/** \file
 *
 *  implementation of RPCMonitorEfficiency class
 *
 *  $Date: 2008/01/22 19:11:44 $
 *  Revision: 1.5 $
 *
 * \author  Camilo Carrillo
 */

#include <DQM/RPCMonitorDigi/interface/RPCMonitorEfficiency.h>

///Log messages
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

///Data Format
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"



#include <cmath>


class DTStationIndex{
public: 
  DTStationIndex():_region(0),_wheel(0),_sector(0),_station(0){}
  DTStationIndex(int region, int wheel, int sector, int station) : 
    _region(region),
    _wheel(wheel),
    _sector(sector),
    _station(station){}
  ~DTStationIndex(){}
  int region() const {return _region;}
  int wheel() const {return _wheel;}
  int sector() const {return _sector;}
  int station() const {return _station;}
  bool operator<(const DTStationIndex& dtind) const{
    if(dtind.region()!=this->region())
      return dtind.region()<this->region();
    else if(dtind.wheel()!=this->wheel())
      return dtind.wheel()<this->wheel();
    else if(dtind.sector()!=this->sector())
      return dtind.sector()<this->sector();
    else if(dtind.station()!=this->station())
      return dtind.station()<this->station();
    return false;
  }
private:
  int _region;
  int _wheel;
  int _sector;
  int _station; 
};


//**********************************************************************************************************



RPCMonitorEfficiency::RPCMonitorEfficiency( const edm::ParameterSet& pset ){
  std::map<RPCDetId, int> buff;
  counter.clear();
  counter.reserve(3);
  //std::cout <<" Buff 1"<<std::endl;
  counter.push_back(buff);
  //std::cout <<" Buff 2"<<std::endl;
  counter.push_back(buff);
  //std::cout <<" Buff 3"<<std::endl;
  counter.push_back(buff);
  totalcounter.clear();
  totalcounter.reserve(3);
  totalcounter[0]=0;
  totalcounter[1]=0;
  totalcounter[2]=0;
  theRecHits4DLabel = pset.getParameter<std::string>("recHits4DLabel");
  digiLabel=pset.getParameter<std::string>("digiLabel");
  EffSaveRootFile  = pset.getUntrackedParameter<bool>("EffSaveRootFile", false); 
  EffSaveRootFileEventsInterval  = pset.getUntrackedParameter<int>("EffEventsInterval", 10000); 
  EffRootFileName  = pset.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiency.root"); 

  /// get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();
  

  dbe->showDirStructure();

  _idList.clear(); 
  //ofrej.open("rejected.txt");
}

void RPCMonitorEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ){
  std::map<RPCDetId, int> buff;
  char layerLabel[128];
  char meIdRPC [128];
  char meIdDT [128];
  float dx=0.,dy=0.,dz=0.,Xo=0.,Yo=0.,X=0.,Y=0.,Z=0.,p1x=0.,p2x=0.,p3x=0.,p4x=0.,p1z=0.,p2z=0.,p3z=0.,p4z=0.,dx3=0.,dy3=0.,dz3=0.,Xo3=0.,Yo3=0.,x3=0.,x4=0.,z3=0.,z4=0.,m3=0.,m4=0.,xc=0.,zc=0.,b3=0.,b4=0.,w3=0.,w4=0.;

  float widestrip=5.;
  float widestripsRB4=8.;
  float circError=3.;
  float angle=0.01;

  edm::ESHandle<DTGeometry> dtGeo;
  iSetup.get<MuonGeometryRecord>().get(dtGeo);
  
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  iEvent.getByLabel(theRecHits4DLabel, all4DSegments);
  
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel(digiLabel, rpcDigis);

  std::map<DTStationIndex,std::set<RPCDetId> > rollstore;
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    RPCRoll* ir = dynamic_cast<RPCRoll*>(*it);
    RPCDetId rpcId = ir->id();
    int region=rpcId.region();
    int wheel=rpcId.ring();
    int sector=rpcId.sector();
    int station=rpcId.station();
    
    DTStationIndex ind(region,wheel,sector,station);
    
    std::set<RPCDetId> myrolls;
    if (rollstore.find(ind)!=rollstore.end()){
      myrolls=rollstore[ind];
    }
    myrolls.insert(rpcId);
    rollstore[ind]=myrolls;
  }

  
  if(all4DSegments->size()>0){
    //std::cout<<"Number of Segments in this event = "<<all4DSegments->size()<<std::endl;
    

    std::map<DTChamberId,int> scounter;
    DTRecSegment4DCollection::const_iterator segment;  

    
    for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
      scounter[segment->chamberId()]++;
    }    
    
    //std::cout<<"Loop over all the 4D Segments"<<std::endl;
    //loop over all the 4D Segments

    for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
      DTChamberId DTId = segment->chamberId();
      //std::cout<<"\t This Segment is in Chamber id: "<<DTId<<std::endl;
      //std::cout<<"\t Number of segments in this DT = "<<scounter[DTId]<<std::endl;
      //std::cout<<"\t DT Segment Dimension "<<segment->dimension()<<std::endl; 
      //std::cout<<"\t Is the only in this DT?"<<std::endl;

      //there must be only one segment per Chamber
      if(scounter[DTId] == 1){	
	//std::cout<<"\t \t yes"<<std::endl;
	int dtWheel = DTId.wheel();
	int dtStation = DTId.station();
	int dtSector = DTId.sector();
	
	LocalPoint segmentPosition= segment->localPosition();
	LocalVector segmentDirection=segment->localDirection();

	const GeomDet* gdet=dtGeo->idToDet(segment->geographicalId());
	const BoundPlane & DTSurface = gdet->surface();

        //check if the dimension of the segment is 4 

	if(segment->dimension()==4){
	  Xo=segmentPosition.x();
	  Yo=segmentPosition.y();
	  dx=segmentDirection.x();
	  dy=segmentDirection.y();
	  dz=segmentDirection.z();
	  //std::cout<<"\t \t Loop over all the rolls asociated to this DT"<<std::endl;
	  std::set<RPCDetId> rollsForThisDT = 
	    rollstore[DTStationIndex(0,dtWheel,dtSector,dtStation)];
	  //Loop over all the rolls
	  for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
	    const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
	    //To get the roll's surface
	    const BoundPlane & RPCSurface = rollasociated->surface(); 
	    //std::cout<<"\t \t RollID: "<<rollasociated->id()<<std::endl;
	    //std::cout<<"\t \t Doing the extrapolation"<<std::endl;
	    //std::cout<<"\t \t DT Segment Direction in DTLocal "<<segmentDirection<<std::endl;
	    //std::cout<<"\t \t DT Segment Point in DTLocal "<<segmentPosition<<std::endl;
	    
	    GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
	    //std::cout<<"\t \t Center (0,0,0) of the Roll in Global"<<CenterPointRollGlobal<<std::endl;
	    
	    LocalPoint CenterRollinDTFrame = DTSurface.toLocal(CenterPointRollGlobal);
	    //std::cout<<"\t \t Center (0,0,0) Roll In DTLocal"<<CenterRollinDTFrame<<std::endl;
	    
	    float D=CenterRollinDTFrame.z();
	    //std::cout<<"\t \t D="<<D<<"cm"<<std::endl;
	    
	    X=Xo+dx*D/dz;
	    Y=Yo+dy*D/dz;
	    Z=D;
	    
	    const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&(rollasociated->topology()));
	    LocalPoint xmin = top_->localPosition(0.);
	    LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
	    float rsize = fabs( xmax.x()-xmin.x() )*0.5;
	    float stripl = top_->stripLength();
	  	  
	    //std::cout<<"\t \t X Predicted in DTLocal= "<<X<<"cm"<<std::endl;
	    //std::cout<<"\t \t Y Predicted in DTLocal= "<<Y<<"cm"<<std::endl;
	    //std::cout<<"\t \t Z Predicted in DTLocal= "<<Z<<"cm"<<std::endl;
	    
	    GlobalPoint GlobalPointExtrapolated = 
	      DTSurface.toGlobal(LocalPoint(X,Y,Z));
	    //std::cout<<"\t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
	    
	    LocalPoint PointExtrapolatedRPCFrame =
	      RPCSurface.toLocal(GlobalPointExtrapolated);
	    //std::cout<<"\t \t Point Extrapolated in RPCLocal"<<PointExtrapolatedRPCFrame<< std::endl;
	    //std::cout<<"\t \t Does the extrapolation go inside this roll?"<<std::endl;
	    //conditions to find the right roll to extrapolate
	    
	    if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01 && fabs(PointExtrapolatedRPCFrame.x()) < rsize && fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
	      //std::cout<<"\t \t \t yes"<<std::endl;	
	      //getting the number of the strip
	      const float stripPredicted = 
		rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
	      
	      //std::cout<<"\t \t \t Candidate"<<rollasociated->id()<<" "<<"(from DT Segment) STRIP---> "<<stripPredicted<< std::endl;
	      
	      //--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
	      
	      RPCDetId  rollId = rollasociated->id();
	      uint32_t id = rollId.rawId();

	      RPCDetId otherRollId1,otherRollId2;

	      if(rollId.roll() == 1){
		RPCDetId tempRollId1(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),2);
		RPCDetId tempRollId2(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),3);
		otherRollId1 = tempRollId1;
		otherRollId2 = tempRollId2;
	      }
	      else if(rollId.roll() == 2){

		RPCDetId tempRollId1(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),1);
		RPCDetId tempRollId2(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),3);
		otherRollId1 = tempRollId1;
		otherRollId2 = tempRollId2;
	      }
	      else if(rollId.roll() == 3){
		RPCDetId tempRollId1(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),1);
		RPCDetId tempRollId2(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),2);
		otherRollId1 = tempRollId1;
		otherRollId2 = tempRollId2;
	      }

	      RPCDigiCollection::Range rpcRangeDigi1=rpcDigis->get(otherRollId1);
	      RPCDigiCollection::Range rpcRangeDigi2=rpcDigis->get(otherRollId2);

	      _idList.push_back(id);
	      
	      char detUnitLabel[128];
	      sprintf(detUnitLabel ,"%d",id);
	      sprintf(layerLabel ,"layer%d_subsector%d_roll%d",rollId.layer(),rollId.subsector(),rollId.roll());
	      
	      std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
	      if (meItr == meCollection.end()){
		meCollection[id] = bookDetUnitMEEff(rollId);
		//std::cout << "\t \t \t Create new histograms  for "<<layerLabel<<std::endl;
	      }
	      
	      std::map<std::string, MonitorElement*> meMap=meCollection[id];
	      sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
	      meMap[meIdDT]->Fill(stripPredicted);

	      sprintf(meIdDT,"ExpectedOccupancy2DFromDT_%s",detUnitLabel);
	      meMap[meIdDT]->Fill(stripPredicted,Y);

	      //std::cout << "\t \t \t One for counterPREDICT"<<std::endl;
	      totalcounter[0]++;
	      buff=counter[0];
	      buff[rollId]++;
	      counter[0]=buff;
	      //-------------------------------------------------------------------
	      
	      //std::cout<<"\t \t \t We have a Candidate let's see in the digis!"<<std::endl;
	      
	      bool anycoincidence=false;
	      int stripDetected = 0;
	      RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());

	      int stripCounter = 0;

	      for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){//loop over the digis in the event
		stripCounter++;
		//std::cout<<"\t \t \t \t Digi "<<*digiIt<<std::endl;//print the digis in the event
		stripDetected=digiIt->strip();

		float res = (float)(stripDetected) - stripPredicted;
		sprintf(meIdRPC,"RPCResiduals_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(res);
		
		sprintf(meIdRPC,"RPCResiduals2D_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(res,Y);
		
		if(res > 7)
		  std::cout<<"STRANGE     "<<"EVENTO NUM = "<<iEvent.id().event()<<"   Residuo = "<<res<<"   Strip Num = "<<stripDetected<<"   Strip totali = "<<stripCounter<<std::endl;
		//compare the strip Detected with the predicted
		if(fabs((float)(stripDetected) - stripPredicted) < widestrip){
		  //std::cout <<"\t \t \t \t COINCEDENCE Predict "
		  //    <<stripPredicted<<" Detect "
		  //    <<stripDetected<<std::endl;
		  anycoincidence=true;
		  //break;//funciona solo para hacerlo mas rapido
		  //We can not divide two diferents things
		}
	      }
	      if (anycoincidence) {
		sprintf(meIdRPC,"ExpectedOccupancyFromDT_forCrT_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(stripPredicted);
		sprintf(meIdRPC,"RealDetectedOccupancy_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(stripDetected);
		sprintf(meIdRPC,"YExpectedOccupancyFromDT_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(Y);

		for (RPCDigiCollection::const_iterator digiIt1 = rpcRangeDigi1.first;digiIt1!=rpcRangeDigi1.second;++digiIt1){

		  if(fabs(stripDetected - digiIt1->strip()) <= 1){
		    
		    sprintf(meIdRPC,"XCrossTalk_1_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(stripPredicted);

		    sprintf(meIdRPC,"XDetectCrossTalk_1_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(stripDetected);

		    sprintf(meIdRPC,"YCrossTalk_1_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(Y);
		    break;
		  }
		}

		for (RPCDigiCollection::const_iterator digiIt2 = rpcRangeDigi2.first;digiIt2!=rpcRangeDigi2.second;++digiIt2){

		  if(fabs(stripDetected - digiIt2->strip()) <= 1){
		    
		    sprintf(meIdRPC,"XCrossTalk_2_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(stripPredicted);

		    sprintf(meIdRPC,"XDetectCrossTalk_2_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(stripDetected);

		    sprintf(meIdRPC,"YCrossTalk_2_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(Y);
		    break;
		  }
		}

		sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(stripPredicted);

		sprintf(meIdRPC,"RPCDataOccupancy2D_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(stripPredicted,Y);

		totalcounter[1]++;
		buff=counter[1];
		buff[rollId]++;
		counter[1]=buff;		
	      }
	      else {
		//std::cout <<"\t \t \t \t XXXXX THIS PREDICTION DOESN'T HAVE ANY CORRESPONDENCE WITH THE DATA"<<std::endl;
		totalcounter[2]++;
		buff=counter[2];
		buff[rollId]++;
		counter[2]=buff;		
		//std::cout << "\t \t \t \t One for counterFAIL"<<std::endl;
		//ofrej<<"Wh "<<dtWheel<<"\t  St "<<dtStation
		//     <<"\t Se "<<dtSector<<"\t Event "
		//    <<iEvent.id().event()<<std::endl;
	      }
	    }
	    else {
	      //std::cout<<"\t \t \t no"<<std::endl;
	    }//Condition for the right match
	  }//loop over all the rolls


	  // dedicated RB4 analysis part, that misses DT 4D segments


	}else if(segment->dimension()==2&&dtStation==4){
	  
	  LocalVector segmentDirectionMB4=segmentDirection;
	  LocalPoint segmentPositionMB4=segmentPosition;
	  

	  //std::cout<<"\t \t 2D in RB4"<<DTId<<" with D="<<segment->dimension()<<localDirection<<segmentPositionMB4<<std::endl;	  
	  bool compatiblesegments=false;
	  Xo=segmentPositionMB4.x();
	  dx=segmentDirectionMB4.x();
	  dz=segmentDirectionMB4.z();
	  //std::cout<<"\t \t Loop over all the segments"<<std::endl;	  
	  DTRecSegment4DCollection::const_iterator segMB3;  

	  const BoundPlane& DTSurface4 = dtGeo->idToDet(DTId)->surface();
	  w4 = DTSurface4.bounds().thickness()*0.5; // along local Z

	  for(segMB3=all4DSegments->begin();segMB3!=all4DSegments->end();++segMB3){
	    DTChamberId dtid = segMB3->chamberId();
	    if(dtid.station()==3){
	      const GeomDet* gdet3=dtGeo->idToDet(segMB3->geographicalId());
	      const BoundPlane & DTSurface3 = gdet3->surface();
	      w3 = DTSurface3.bounds().thickness()*0.5; // along local Z
	      
	      dx3=segMB3->localDirection().x();
	      dy3=segMB3->localDirection().y();
	      dz3=segMB3->localDirection().z();
	      
	      //LocalVector segDirMB4inMB3Frame=DTSurface3.toLocal(DTSurface4.toGlobal(segmentDirectionMB4));

	      if(fabs(dx-dx3)<=angle&&fabs(dz-dz3)<=angle){//same direction?
		compatiblesegments=true;
	      }else{
		//They don't have the same local dir, and the segments in diferent sectors,compatibles with circle?
		Xo3=segMB3->localPosition().x();
		Yo3=segMB3->localPosition().y();

		//Do we have segments compatibles with a muon?
		
		// pos=seg.position()
		// dir=seg.direction()
		// dest =pos + dir*height*cos(dir.theta())
		
		//std::cout<<"Ancho de MB3 w3="<<w3<<" |Ancho de MB4 w4= "<<w4<<std::endl;
		//std::cout<<"Informacion Inicial MB4"<<segment->localDirection()<<" "<<segment->localPosition()<<std::endl;
		//std::cout<<"Informacion Inicial MB3"<<segMB3->localDirection()<<" "<<segMB3->localPosition()<<std::endl;
		compatiblesegments=false;
		
		//This in MB3 Local
		
		p1x=Xo3+dx3*w3/dz3;
		p1z=w3;
		
		p2x=Xo3-dx3*w3/dz3;
		p2z=-w3;
		
		//This in MB4 Local
		
		p3x=Xo+dx*w4/dz;
		p3z=w4;
		
		p4x=Xo-dx*w4/dz;
		p4z=-w4;
		
		LocalPoint P1=LocalPoint(p1x,0,p1z);
		LocalPoint P2=LocalPoint(p2x,0,p2z);
		LocalPoint P3=LocalPoint(p3x,0,p3z);
		LocalPoint P4=LocalPoint(p4x,0,p4z);
		
		//std::cout<<"Points in MB3="<<P1<<" --- "<<P2<<std::endl;
		//std::cout<<"Points in MB4="<<P3<<" --- "<<P4<<std::endl;
		
		//Now we have to convert to global
		
		LocalPoint P1g=P1;
		LocalPoint P2g=P2;
		
		LocalPoint P3g=DTSurface3.toLocal(DTSurface.toGlobal(P3));
		LocalPoint P4g=DTSurface3.toLocal(DTSurface.toGlobal(P4));
		
		//std::cout<<"Points in MB3 ="<<P1g<<" --- "<<P2g<<std::endl;
		//std::cout<<"Points in MB4 in MB3Frame="<<P3g<<" --- "<<P4g<<std::endl;
		
		
		float dx3g=P1g.x()-P2g.x();
		float dz3g=P1g.z()-P2g.z();
	      
		float dxg=P3g.x()-P4g.x();
		float dzg=P3g.z()-P4g.z();
		
		//std::cout<<"dx3g="<<dx3g<<" dz3g="<<dz3g<<std::endl;
		//std::cout<<"dxg="<<dxg<<" dzg="<<dzg<<std::endl;
		
		m3=-dx3g/dz3g;
		m4=-dxg/dzg;
		
		x3=(P1g.x()+P2g.x())*0.5;
		z3=(P1g.z()+P2g.z())*0.5;
		
		x4=(P3g.x()+P4g.x())*0.5;
		z4=(P3g.z()+P4g.z())*0.5;
		
		b3=z3-m3*x3;
		b4=z4-m4*x4;
		
		if(m3!=m4){
		  xc=(b4-b3)/(m3-m4);
		  zc=m3*xc+b3;
		
		  GlobalPoint Pc=GlobalPoint(xc,0,zc);
		  
		  //std::cout<<Pc<<std::endl;
		  
		  float distance=fabs((GlobalPoint(P2g.x(),0,P2g.z())-GlobalPoint(Pc.x(),0,Pc.z())).mag()-(GlobalPoint(P3g.x(),0,P3g.z())-GlobalPoint(Pc.x(),0,Pc.z())).mag());
		  
		  if(distance<circError){
		    compatiblesegments=true;
		    //std::cout<<"YES in the same circle... p2 a C="<<P1g<<P2g<<P3g<<P4g<<"Distancia "<<distance<<std::endl;
		  }
		  else{
		    //std::cout<<"NOT in the same circle... p2 a C="<<P1g<<P2g<<P3g<<P4g<<"Distancia "<<distance<<std::endl;
		    compatiblesegments=false;
		  }
		}
		else{
		  //std::cout<<"We have the same slope m3="<<m3<<" m4="<<m4<<std::endl;
		  if(fabs(b3-b4)<=circError){
		    compatiblesegments=true;
		    //std::cout<<"and the segments are in a line"<<std::endl;
		  }else{std::cout<<"But we don't have the same intercept b3="<<b3<<" b4="<<b4<<std::endl;}
		  compatiblesegments=false;
		  system("sleep 30");
		}
	      }
		
	      //conditions in MB3
	      if(scounter[dtid]==1 && compatiblesegments){
		//std::cout<<"********\t \t \t In the same event there is a segment in RB3 "<<dtid<<" with D="<<segMB3->dimension()<<segMB3->localDirection()<<segMB3->localPosition()<<"scounter "<<scounter[dtid]<<std::endl;
		
		std::set<RPCDetId> rollsForThisDT = 
		  rollstore[DTStationIndex(0,dtWheel,dtSector,dtStation)];
		//Loop over all the rolls asociated to RB4
		for (std::set<RPCDetId>::iterator iteraRoll=rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
		  const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
		  //To get the roll's surface
		  const BoundPlane & RPCSurfaceRB4 = rollasociated->surface(); 
		  
		  const GeomDet* gdet=dtGeo->idToDet(segMB3->geographicalId());
		  const BoundPlane & DTSurfaceMB3 = gdet->surface();
		  
		  //std::cout<<"\t \t \t RollID: should be RB4"<<rollasociated->id()<<std::endl;
		  //std::cout<<"\t \t \t Making the extrapolation"<<std::endl;
		  //std::cout<<"\t \t \t DT Segment Direction in MB3 DTLocal "<<segMB3->localDirection()<<std::endl;
		  //std::cout<<"\t \t \t DT Segment Point in MB3 DTLocal "<<segMB3->localPosition()<<std::endl;
		  
		  GlobalPoint CenterPointRollGlobal=RPCSurfaceRB4.toGlobal(LocalPoint(0,0,0));
		  //std::cout<<"\t \t \t Center (0,0,0) of the RB4 Roll in Global"<<CenterPointRollGlobal<<std::endl;
		  
		  LocalPoint CenterRollinDTFrame = DTSurfaceMB3.toLocal(CenterPointRollGlobal);
		  //std::cout<<"\t \t \t Center (0,0,0) Roll In DT MB3 Local"<<CenterRollinDTFrame<<std::endl;
		  
		  float D=CenterRollinDTFrame.z();
		  //std::cout<<"\t \t \t D="<<D<<"cm"<<std::endl;
		  
		  X=Xo3+dx3*D/dz3;
		  Y=Yo3+dy3*D/dz3;
		  Z=D;
		  
		  const RectangularStripTopology* top_=dynamic_cast<const RectangularStripTopology*>(&(rollasociated->topology()));
		  LocalPoint xmin = top_->localPosition(0.);
		  LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
		  float rsize = fabs( xmax.x()-xmin.x() )*0.5;
		  float stripl = top_->stripLength();
		  
		  //std::cout<<"\t \t \t X Predicted in DT MB3 Local= "<<X<<"cm"<<std::endl;
		  //std::cout<<"\t \t \t Y Predicted in DT MB3 Local= "<<Y<<"cm"<<std::endl;
		  //std::cout<<"\t \t \t Z Predicted in DT MB3 Local= "<<Z<<"cm"<<std::endl;
		  
		  GlobalPoint GlobalPointExtrapolated = DTSurfaceMB3.toGlobal(LocalPoint(X,Y,Z));
		  //std::cout<<"\t \t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
		  
		  LocalPoint PointExtrapolatedRPCFrame = RPCSurfaceRB4.toLocal(GlobalPointExtrapolated);
		  //std::cout<<"\t \t \t Point Extrapolated in RPC RB4 Local"<<PointExtrapolatedRPCFrame<< std::endl;
		  
		  //std::cout<<"\t \t \t Does the extrapolation go inside this roll?"<<std::endl;
		  //conditions to find the right roll to extrapolate
		  if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01  &&
		     fabs(PointExtrapolatedRPCFrame.x()) < rsize &&
		     fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
		    
		    //std::cout<<"\t \t \t \t yes"<<std::endl;
		    //getting the number of the strip
		    const float stripPredicted=
		      rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		    
		    //std::cout<<"\t \t \t \t Candidate"<<rollasociated->id()<<" "<<"(from DT Segment) STRIP---> "<<stripPredicted<< std::endl;
		    
		    //--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
		    
		    RPCDetId  rollId = rollasociated->id();
		    uint32_t id = rollId.rawId();
		    
		    RPCDetId otherRollId1,otherRollId2;
		    
		    if(rollId.roll() == 1){
		      RPCDetId tempRollId1(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),2);
		      RPCDetId tempRollId2(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),3);
		      otherRollId1 = tempRollId1;
		      otherRollId2 = tempRollId2;
		    }
		    else if(rollId.roll() == 2){
		      
		      RPCDetId tempRollId1(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),1);
		      RPCDetId tempRollId2(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),3);
		      otherRollId1 = tempRollId1;
		      otherRollId2 = tempRollId2;
		    }
		    else if(rollId.roll() == 3){
		      RPCDetId tempRollId1(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),1);
		      RPCDetId tempRollId2(rollId.region(),rollId.ring(),rollId.station(),rollId.sector(),rollId.layer(),rollId.subsector(),2);
		      otherRollId1 = tempRollId1;
		      otherRollId2 = tempRollId2;
		    }
		    RPCDigiCollection::Range rpcRangeDigi1=rpcDigis->get(otherRollId1);
		    RPCDigiCollection::Range rpcRangeDigi2=rpcDigis->get(otherRollId2);

		    _idList.push_back(id);
		    
		    char detUnitLabel[128];
		    sprintf(detUnitLabel ,"%d",id);
		    sprintf(layerLabel ,"layer%d_subsector%d_roll%d",rollId.layer(),rollId.subsector(),rollId.roll());
		    
		    std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
		    if (meItr == meCollection.end()){
		      meCollection[id] = bookDetUnitMEEff(rollId);
		      std::cout << "\t \t \t \t Create new histograms  for "<<layerLabel<<std::endl;
		    }
		    
		    std::map<std::string, MonitorElement*> meMap=meCollection[id];
		    sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
		    meMap[meIdDT]->Fill(stripPredicted);

		    sprintf(meIdDT,"ExpectedOccupancy2DFromDT_%s",detUnitLabel);
		    meMap[meIdDT]->Fill(stripPredicted,Y);

		    //std::cout << "\t \t \t \t One for counterPREDICT"<<std::endl;
		    totalcounter[0]++;
		    buff=counter[0];
		    buff[rollId]++;
		    counter[0]=buff;		
		    
		    bool anycoincidence=false;
		    int stripDetected = 0;
		    RPCDigiCollection::Range rpcRangeDigi = 
		      rpcDigis->get(rollasociated->id());
		    		   
		    int stripCounter = 0;

		    //loop over the digis in the event
		    for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
		      stripCounter++;
		      //std::cout<<"\t \t \t \t \t Digi "<<*digiIt<<std::endl;//print the digis in the event
		      stripDetected=digiIt->strip();

		      stripCounter++;
		      float res = (float)(stripDetected) - stripPredicted;
		      sprintf(meIdRPC,"RPCResiduals_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(res);
		
		      sprintf(meIdRPC,"RPCResiduals2D_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(res,Y);
		      
		      if(res > 7)
			std::cout<<"STRANGE     "<<"EVENTO NUM = "<<iEvent.id().event()<<"   Residuo = "<<res<<"   Strip Num = "<<stripDetected<<"   Strip totali = "<<stripCounter<<std::endl;
		      if(fabs((float)(stripDetected) - stripPredicted)<widestripsRB4){//Detected Vs Predicted
			//std::cout <<"\t \t \t \t \t COINCEDENCE Predict "<<stripPredicted<<" Detect "<<stripDetected<<std::endl;
			anycoincidence=true;
			//break;//funciona solo para hacerlo mas rapido
		      }
		    }
		    if (anycoincidence){

		      sprintf(meIdRPC,"ExpectedOccupancyFromDT_forCrT_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(stripPredicted);
		      sprintf(meIdRPC,"RealDetectedOccupancy_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(stripDetected);
		      sprintf(meIdRPC,"YExpectedOccupancyFromDT_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(Y);

		      for (RPCDigiCollection::const_iterator digiIt1 = rpcRangeDigi1.first;digiIt1!=rpcRangeDigi1.second;++digiIt1){

			if(fabs(stripDetected - digiIt1->strip()) <= 1){
			  
			  sprintf(meIdRPC,"XCrossTalk_1_%s",detUnitLabel);
			  meMap[meIdRPC]->Fill(stripPredicted);
			  
			  sprintf(meIdRPC,"XDetectCrossTalk_1_%s",detUnitLabel);
			  meMap[meIdRPC]->Fill(stripDetected);
			  
			  sprintf(meIdRPC,"YCrossTalk_1_%s",detUnitLabel);
			  meMap[meIdRPC]->Fill(Y);
			  break;
			}
		      }

		      for (RPCDigiCollection::const_iterator digiIt2 = rpcRangeDigi2.first;digiIt2!=rpcRangeDigi2.second;++digiIt2){
			
			if(fabs(stripDetected - digiIt2->strip()) <= 1){
			  
			  sprintf(meIdRPC,"XCrossTalk_2_%s",detUnitLabel);
			  meMap[meIdRPC]->Fill(stripPredicted);
			  
			  sprintf(meIdRPC,"XDetectCrossTalk_2_%s",detUnitLabel);
			  meMap[meIdRPC]->Fill(stripDetected);
			  
			  sprintf(meIdRPC,"YCrossTalk_2_%s",detUnitLabel);
			  meMap[meIdRPC]->Fill(Y);
			  break;
			}
		      }

		      sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(stripPredicted);

		      sprintf(meIdRPC,"RPCDataOccupancy2D_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(stripPredicted,Y);

		      totalcounter[1]++;
		      buff=counter[1];
		      buff[rollId]++;
		      counter[1]=buff;		
		    }
		    else{
		      totalcounter[2]++;
		      buff=counter[2];
		      buff[rollId]++;
		      counter[2]=buff;		
		      //std::cout <<"\t \t \t \t \t XXXXX THIS PREDICTION DOESN'T HAVE ANY CORRESPONDENCE WITH THE DATA"<<std::endl;
		      //std::cout << "\t \t \t \t \t One for counterFAIL"<<std::endl;
		      //ofrej<<"Wh "<<dtWheel<<"\t St "<<dtStation
		      //	 <<"\t Se "<<dtSector<<"\t Event "
		      //	 <<iEvent.id().event()<<std::endl;
		      
		    }
		  }
		  else{
		    //std::cout<<"\t \t \t \t no"<<std::endl;
		  }//Condition for the right match
		  
		}//loop over all the rolls FOR RB3 -------------------------------
	      }//Si tenemos solo un segmento en la chamber
	    }//Condition Segment in MB3
	  }//Loop avoer all the segments to see if it is the right MB3
	}
	else{
	  //std::cout<<"\t \t Strange Segment"<<std::endl;
	}//Is not a 4D Segment neither a 2D in MB4
      }
      else {
	//std::cout<<"\t \t no"<<std::endl;
      }//There is one segment in the chamber?
    }//loop over the segments
  }else {
    //std::cout<<"This Event doesn't have any DT4DSegment"<<std::endl;
  }//is ther more than 1 segment in this event?
  
}

void RPCMonitorEfficiency::endJob(void){
  std::map<RPCDetId, int> pred = counter[0];
  std::map<RPCDetId, int> obse = counter[1];
  std::map<RPCDetId, int> reje = counter[2];

  std::map<RPCDetId, int>::iterator irpc;
  for (irpc=pred.begin(); irpc!=pred.end();irpc++){
    RPCDetId id=irpc->first;
    int p=pred[id]; 
    int o=obse[id]; 
    int r=reje[id]; 
    assert(p==o+r);
    float ef = float(o)/float(p);
    float er = sqrt(ef*(1.-ef)/float(p));
    std::cout <<"\n "<<id<<"\t Predicted "<<p<<"\t Observed "<<o<<"\t Eff = "<<ef*100.<<" % +/- "<<er*100.<<" %";
    if(ef<0.8){
      std::cout<<"\t \t Warning!";
    } 
  }
  
  float tote = float(totalcounter[1])/float(totalcounter[0]);
  float totr = sqrt(tote*(1.-tote)/float(totalcounter[0]));
  std::cout <<"\n\n \t \t TOTAL EFFICIENCY \t Predicted "<<totalcounter[1]<<"\t Observed "<<totalcounter[0]<<"\t Eff = "<<tote*100.<<"\t +/- \t"<<totr*100.<<"%"<<std::endl;
  std::cout <<totalcounter[1]<<" "<<totalcounter[0]<<" flagcode"<<std::endl;

  
  std::vector<uint32_t>::iterator meIt;
  for(meIt = _idList.begin(); meIt != _idList.end(); ++meIt){

    char detUnitLabel[128];
    char meIdRPC [128];
    char meIdDT [128];
    char meIdDTCrT [128];
    char effIdRPC [128];
    char meIdRPCCrT [128];

    char XCrTalkId_1 [128];
    char XCrTalkId_2 [128];
    char effXCrTalkId_1 [128];
    char effXCrTalkId_2 [128];

    char XCrTalkDetId_1 [128];
    char XCrTalkDetId_2 [128];
    char effXCrTalkDetId_1 [128];
    char effXCrTalkDetId_2 [128];

    char YCrTalkId_1 [128];
    char YCrTalkId_2 [128];
    char effYCrTalkId_1 [128];
    char effYCrTalkId_2 [128];

    char YCrTalkDTId [128];

    char meIdRPC_2D [128];
    char meIdDT_2D [128];
    char effIdRPC_2D [128];

    sprintf(detUnitLabel ,"%d",*meIt);
    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
    sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(meIdDTCrT,"ExpectedOccupancyFromDT_forCrT_%s",detUnitLabel);
    sprintf(YCrTalkDTId,"YExpectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(meIdRPCCrT,"RealDetectedOccupancy_%s",detUnitLabel);

    sprintf(effIdRPC,"EfficienyFromDTExtrapolation_%s",detUnitLabel);

    sprintf(meIdRPC_2D,"RPCDataOccupancy2D_%s",detUnitLabel);
    sprintf(meIdDT_2D,"ExpectedOccupancy2DFromDT_%s",detUnitLabel);
    sprintf(effIdRPC_2D,"EfficienyFromDT2DExtrapolation_%s",detUnitLabel);

    sprintf(XCrTalkId_1,"XCrossTalk_1_%s",detUnitLabel);
    sprintf(XCrTalkId_2,"XCrossTalk_2_%s",detUnitLabel);
    sprintf(YCrTalkId_1,"YCrossTalk_1_%s",detUnitLabel);
    sprintf(YCrTalkId_2,"YCrossTalk_2_%s",detUnitLabel);
    sprintf(XCrTalkDetId_1,"XDetectCrossTalk_1_%s",detUnitLabel);
    sprintf(XCrTalkDetId_2,"XDetectCrossTalk_2_%s",detUnitLabel);

    sprintf(effXCrTalkId_1,"XCrossTalkFromDTExtrapolation_1_%s",detUnitLabel);
    sprintf(effXCrTalkId_2,"XCrossTalkFromDTExtrapolation_2_%s",detUnitLabel);
    sprintf(effXCrTalkDetId_1,"XCrossTalkFromDetectedStrip_1_%s",detUnitLabel);
    sprintf(effXCrTalkDetId_2,"XCrossTalkFromDetectedStrip_2_%s",detUnitLabel);
    sprintf(effYCrTalkId_1,"YCrossTalkFromDTExtrapolation_1_%s",detUnitLabel);
    sprintf(effYCrTalkId_2,"YCrossTalkFromDTExtrapolation_2_%s",detUnitLabel);

    std::map<std::string, MonitorElement*> meMap=meCollection[*meIt];

    for(unsigned int i=1;i<=100;++i){
      
      if(meMap[meIdDT]->getBinContent(i) != 0){
	float eff = meMap[meIdRPC]->getBinContent(i)/meMap[meIdDT]->getBinContent(i);
	float erreff = sqrt(eff*(1-eff)/meMap[meIdDT]->getBinContent(i));
	meMap[effIdRPC]->setBinContent(i,eff*100.);
	meMap[effIdRPC]->setBinError(i,erreff*100.);
      }
    }
    for(unsigned int i=1;i<=100;++i){
      for(unsigned int j=1;j<=200;++j){
	if(meMap[meIdDT_2D]->getBinContent(i,j) != 0){
	  float eff = meMap[meIdRPC_2D]->getBinContent(i,j)/meMap[meIdDT_2D]->getBinContent(i,j);
	  float erreff = sqrt(eff*(1-eff)/meMap[meIdDT_2D]->getBinContent(i,j));
	  meMap[effIdRPC_2D]->setBinContent(i,j,eff*100.);
	  meMap[effIdRPC_2D]->setBinError(i,j,erreff*100.);
	}
      }
    }

    //--------------------  CROSS TALK PLOT ---------------------------------------------------

    //-------------------- With predicted STRIP -----------------------------------------------

    for(unsigned int i=1;i<=100;++i){
      
      if(meMap[meIdDTCrT]->getBinContent(i) != 0){
	float crt = meMap[XCrTalkId_1]->getBinContent(i)/meMap[meIdDTCrT]->getBinContent(i);
	float errcrt = sqrt(crt*(1-crt)/meMap[meIdDTCrT]->getBinContent(i));
	meMap[effXCrTalkId_1]->setBinContent(i,crt*100.);
	meMap[effXCrTalkId_1]->setBinError(i,errcrt*100.);
      }
    }

    for(unsigned int i=1;i<=100;++i){
      
      if(meMap[meIdDTCrT]->getBinContent(i) != 0){
	float crt = meMap[XCrTalkId_2]->getBinContent(i)/meMap[meIdDTCrT]->getBinContent(i);
	float errcrt = sqrt(crt*(1-crt)/meMap[meIdDTCrT]->getBinContent(i));
	meMap[effXCrTalkId_2]->setBinContent(i,crt*100.);
	meMap[effXCrTalkId_2]->setBinError(i,errcrt*100.);
      }
    }

    //-------------------- With detected STRIP -----------------------------------------------

    for(unsigned int i=1;i<=100;++i){
      if(meMap[meIdRPCCrT]->getBinContent(i) != 0){
	float crt = meMap[XCrTalkDetId_1]->getBinContent(i)/meMap[meIdRPCCrT]->getBinContent(i);
	float errcrt = sqrt(crt*(1-crt)/meMap[meIdRPCCrT]->getBinContent(i));
	meMap[effXCrTalkDetId_1]->setBinContent(i,crt*100.);
	meMap[effXCrTalkDetId_1]->setBinError(i,errcrt*100.);
      }
    }

    for(unsigned int i=1;i<=100;++i){
      
      if(meMap[meIdRPCCrT]->getBinContent(i) != 0){
	float crt = meMap[XCrTalkDetId_2]->getBinContent(i)/meMap[meIdRPCCrT]->getBinContent(i);
	float errcrt = sqrt(crt*(1-crt)/meMap[meIdRPCCrT]->getBinContent(i));
	meMap[effXCrTalkDetId_2]->setBinContent(i,crt*100.);
	meMap[effXCrTalkDetId_2]->setBinError(i,errcrt*100.);
      }
    }

    //-------------------- With Y coordinate -------------------------------------------------

    for(unsigned int i=1;i<=200;++i){
      
      if(meMap[YCrTalkDTId]->getBinContent(i) != 0){
	float crt = meMap[YCrTalkId_1]->getBinContent(i)/meMap[YCrTalkDTId]->getBinContent(i);
	float errcrt = sqrt(crt*(1-crt)/meMap[YCrTalkDTId]->getBinContent(i));
	meMap[effYCrTalkId_1]->setBinContent(i,crt*100.);
	meMap[effYCrTalkId_1]->setBinError(i,errcrt*100.);
      }
    }

    for(unsigned int i=1;i<=200;++i){
      
      if(meMap[YCrTalkDTId]->getBinContent(i) != 0){
	float crt = meMap[YCrTalkId_2]->getBinContent(i)/meMap[YCrTalkDTId]->getBinContent(i);
	float errcrt = sqrt(crt*(1-crt)/meMap[YCrTalkDTId]->getBinContent(i));
	meMap[effYCrTalkId_2]->setBinContent(i,crt*100.);
	meMap[effYCrTalkId_2]->setBinError(i,errcrt*100.);
      }
    }
  }

  if(EffSaveRootFile) dbe->save(EffRootFileName);
  //  theFile->Write();
  //  theile->Close();
  std::cout<<"End Job"<<std::endl;
}

RPCMonitorEfficiency::~RPCMonitorEfficiency(){}






