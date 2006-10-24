 /** \file
 *
 *  implementation of RPCMonitorEfficiency class
 *
 *  $Date: 2006/10/14 10:29:28 $
 *  Revision: 1.5 $
 *
 * \author  Camilo Carrillo
 */

#include <DQM/RPCMonitorDigi/interface/RPCMonitorEfficiency.h>

///Log messages
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

//#include "FWCore/Framework/interface/MakerMacros.h" //
#include <FWCore/Framework/interface/ESHandle.h>//

///Data Format
#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"//
#include "DataFormats/MuonDetId/interface/DTChamberId.h"//

#include <Geometry/DTGeometry/interface/DTGeometry.h>//
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>//
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>//
#include <Geometry/CommonDetUnit/interface/GeomDet.h>//
#include <Geometry/Records/interface/MuonGeometryRecord.h>//
#include <Geometry/Surface/interface/LocalError.h>
#include <Geometry/Vector/interface/LocalPoint.h>


#include <cmath>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"




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
  std::cout <<" Buff 1"<<std::endl;
  counter.push_back(buff);
  std::cout <<" Buff 2"<<std::endl;
  counter.push_back(buff);
  std::cout <<" Buff 3"<<std::endl;
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
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  dbe->showDirStructure();

  _idList.clear(); 
  ofrej.open("rejected.txt");
}




void RPCMonitorEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ){
  std::map<RPCDetId, int> buff;
  char layerLabel[128];
  char meIdRPC [128];
  char meIdDT [128];
  float dx=0.,dy=0.,dz=0.,Xo=0.,Yo=0.,X=0.,Y=0.,Z=0.;
  
  float widestrip=5.;
  float widestripsRB4=8.;
  float angledistance=0.5;

  bool inRB1IN[3];
  bool inRB1OUT[3];
  bool inRB2IN[3];
  bool inRB2OUT[3];
  bool inRB3[3];
  bool inRB4[3];
  
  unsigned int i;

  for(i=0;i<3;i++){
    inRB1IN[i]=false;
    inRB1OUT[i]=false;
    inRB2IN[i]=false;
    inRB2OUT[i]=false;
    inRB3[i]=false;
    inRB4[i]=false;
  }
  
  
  edm::ESHandle<DTGeometry> dtGeo;
  iSetup.get<MuonGeometryRecord>().get(dtGeo);
  
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  iEvent.getByLabel(theRecHits4DLabel, all4DSegments);
  
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel(digiLabel, rpcDigis);

  std::map<DTStationIndex,std::set<RPCDetId> > rollstore;
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();
       it<rpcGeo->dets().end();it++){
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
    std::cout<<"Number of Segments in this event = "
	     <<all4DSegments->size()<<std::endl;
    

    std::map<DTChamberId,int> scounter;
    DTRecSegment4DCollection::const_iterator segment;  
    for (segment = all4DSegments->begin();
	 segment!=all4DSegments->end(); ++segment){

//Counting how many segments per DTChamberID 4D do we have in the event, 
      scounter[segment->chamberId()]++;
    }    
    
    std::cout<<"Loop over all the 4D Segments"<<std::endl;
    //loop over all the 4D Segments
    for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
      //check if the dimension of the segment is 4 

      DTChamberId DTId = segment->chamberId();
      std::cout<<"\t This Segment is in Chamber id: "<<DTId<<std::endl;
      std::cout<<"\t Number of segments in this DT = "<<scounter[DTId]<<std::endl;
      std::cout<<"\t DT Segment Dimension "<<segment->dimension()<<std::endl; 
      std::cout<<"\t Is the only in this DT?"<<std::endl;

      //there must be only one segment per Chamber
      if(scounter[DTId] == 1){	
	std::cout<<"\t \t yes"<<std::endl;
	int dtWheel = DTId.wheel();
	int dtStation = DTId.station();
	int dtSector = DTId.sector();
	
	LocalPoint localPoint= segment->localPosition();
	LocalVector localDirection=segment->localDirection();

	const GeomDet* gdet=dtGeo->idToDet(segment->geographicalId());
	const BoundPlane & DTSurface = gdet->surface();

	if(segment->dimension()==4){
	  Xo=localPoint.x();
	  Yo=localPoint.y();
	  dx=localDirection.x();
	  dy=localDirection.y();
	  dz=localDirection.z();
	  std::cout<<"\t \t Loop over all the rolls asociated to this DT"<<std::endl;
	  std::set<RPCDetId> rollsForThisDT = 
	    rollstore[DTStationIndex(0,dtWheel,dtSector,dtStation)];
	  //Loop over all the rolls
	  for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisDT.begin();
	       iteraRoll != rollsForThisDT.end(); iteraRoll++){
	    const RPCRoll* rollasociated = 
	      dynamic_cast<const RPCRoll*> (rpcGeo->idToDetUnit(*iteraRoll));
	    //To get the roll's surface
	    const BoundPlane & RPCSurface = rollasociated->surface(); 
	    std::cout<<"\t \t RollID: "<<rollasociated->id()<<std::endl;
	    std::cout<<"\t \t Making the extrapolation"<<std::endl;
	    std::cout<<"\t \t DT Segment Direction in DTLocal "
		     <<localDirection<<std::endl;
	    std::cout<<"\t \t DT Segment Point in DTLocal "<<localPoint
		     <<std::endl;
	    
	    GlobalPoint CenterPointRollGlobal = 
	      RPCSurface.toGlobal(LocalPoint(0,0,0));
	    std::cout<<"\t \t Center (0,0,0) of the Roll in Global"
		     <<CenterPointRollGlobal<<std::endl;
	    
	    LocalPoint CenterRollinDTFrame = 
	      DTSurface.toLocal(CenterPointRollGlobal);
	    std::cout<<"\t \t Center (0,0,0) Roll In DTLocal"
		     <<CenterRollinDTFrame<<std::endl;
	    
	    float D=CenterRollinDTFrame.z();
	    std::cout<<"\t \t D="<<D<<"cm"<<std::endl;
	    
	    X=Xo+dx*D/dz;
	    Y=Yo+dy*D/dz;
	    Z=D;
	    
	    const RectangularStripTopology* top_= 
	      dynamic_cast<const RectangularStripTopology*>
	      (&(rollasociated->topology()));
	    LocalPoint xmin = top_->localPosition(0.);
	    LocalPoint xmax = 
	      top_->localPosition((float)rollasociated->nstrips());
	    float rsize = fabs( xmax.x()-xmin.x() )*0.5;
	    float stripl = top_->stripLength();
	  	  
	    std::cout<<"\t \t X Predicted in DTLocal= "<<X<<"cm"<<std::endl;
	    std::cout<<"\t \t Y Predicted in DTLocal= "<<Y<<"cm"<<std::endl;
	    std::cout<<"\t \t Z Predicted in DTLocal= "<<Z<<"cm"<<std::endl;
	    
	    GlobalPoint GlobalPointExtrapolated = 
	      DTSurface.toGlobal(LocalPoint(X,Y,Z));
	    std::cout<<"\t \t Point ExtraPolated in Global"
		     <<GlobalPointExtrapolated<< std::endl;
	    
	    LocalPoint PointExtrapolatedRPCFrame =
	      RPCSurface.toLocal(GlobalPointExtrapolated);
	    std::cout<<"\t \t Point Extrapolated in RPCLocal"
		     <<PointExtrapolatedRPCFrame<< std::endl;
	    
	    std::cout<<"\t \t Does the extrapolation go inside this roll?"
		     <<std::endl;
	    
	    //conditions to find the right roll to extrapolate
	    if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01  && 
	       fabs(PointExtrapolatedRPCFrame.x()) < rsize &&
	       fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
	      
	      std::cout<<"\t \t \t yes"<<std::endl;	
	      //getting the number of the strip
	      const float stripPredicted = 
		rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
	      
	      std::cout<<"\t \t \t Candidate"<<rollasociated->id()<<" "<<"(from DT Segment) STRIP---> "<<stripPredicted<< std::endl;
	      
	      //--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
	      
	      RPCDetId  rollId = rollasociated->id();
	      uint32_t id = rollId.rawId();
	      
	      _idList.push_back(id);
	      
	      char detUnitLabel[128];
	      sprintf(detUnitLabel ,"%d",id);
	      sprintf(layerLabel ,"layer%d_subsector%d_roll%d",rollId.layer(),rollId.subsector(),rollId.roll());
	      
	      std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
	      if (meItr == meCollection.end()){
		meCollection[id] = bookDetUnitMEEff(rollId);
		std::cout << "\t \t \t Create new histograms  for "<<layerLabel<<std::endl;
	      }
	      
	      std::map<std::string, MonitorElement*> meMap=meCollection[id];
	      sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
	      meMap[meIdDT]->Fill(stripPredicted);
	      std::cout << "\t \t \t One for counterPREDICT"<<std::endl;
	      totalcounter[0]++;
	      buff=counter[0];
	      buff[rollId]++;
	      counter[0]=buff;
	      //-------------------------------------------------------------------
	      
	      std::cout<<"\t \t \t We have a Candidate let's see in the digis!"<<std::endl;
	      
	      bool anycoincidence=false;
	      int stripDetected = 0;
	      RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());
	    
	      
	      for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){//loop over the digis in the event
		std::cout<<"\t \t \t \t Digi "<<*digiIt<<std::endl;//print the digis in the event
		stripDetected=digiIt->strip();
		//compare the strip Detected with the predicted
		if(fabs((float)(stripDetected) - stripPredicted)<widestrip){
		  std::cout <<"\t \t \t \t COINCEDENCE Predict "
			    <<stripPredicted<<" Detect "
			    <<stripDetected<<std::endl;
		  anycoincidence=true;
		  //We can not divide two diferents things
		}
	      }
	      if (anycoincidence) {
		sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
		meMap[meIdRPC]->Fill(stripPredicted);
		totalcounter[1]++;
		buff=counter[1];
		buff[rollId]++;
		counter[1]=buff;		
	      }
	      else {
		std::cout <<"\t \t \t \t XXXXX THIS PREDICTION DOESN'T HAVE ANY CORRESPONDENCE WITH THE DATA"<<std::endl;
		totalcounter[2]++;
		buff=counter[2];
		buff[rollId]++;
		counter[2]=buff;		
		std::cout << "\t \t \t \t One for counterFAIL"<<std::endl;
		ofrej<<"Wh "<<dtWheel<<" | St "<<dtStation
		     <<"  | Se "<<dtSector<<" | Event "
		     <<iEvent.id().event()<<std::endl;
	      }
	    }
	    else {
	      std::cout<<"\t \t \t no"<<std::endl;
	    }//Condition for the right match
	  }//loop over all the rolls
	  // dedicated RB4 analysis part, that misses DT 4D segments
	}else if(segment->dimension()==2&&dtStation==4){
	  std::cout<<"\t \t 2D in RB4"<<DTId<<" with D="<<segment->dimension()<<localDirection<<localPoint<<std::endl;	  
	  bool correspondenceinRB3=false;
	  Xo=localPoint.x();
	  dx=localDirection.x();
	  dz=localDirection.z();
	  std::cout<<"\t \t Loop over all the segments"<<std::endl;	  
	  DTRecSegment4DCollection::const_iterator segMB3;  
	  for(segMB3=all4DSegments->begin();
	      segMB3!=all4DSegments->end();++segMB3){
	    DTChamberId dtid = segMB3->chamberId();
	    float dx3=segMB3->localDirection().x();
	    float dy3=segMB3->localDirection().y();
	    float dz3=segMB3->localDirection().z();
	    float Xo3=segMB3->localPosition().x();
	    float Yo3=segMB3->localPosition().y();
	    //conditions in MB3
	    if(dtid.station()==3 && 
	       scounter[dtid]==1 && 
	       fabs(dx-dx3) < angledistance &&
	       fabs(dz-dz3) < angledistance){
	      std::cout<<"********\t \t \t In the same event there is a segment in RB3 "<<dtid<<" with D="<<segMB3->dimension()<<segMB3->localDirection()<<segMB3->localPosition()<<"scounter "<<scounter[dtid]<<std::endl;
	      
	      std::set<RPCDetId> rollsForThisDT = 
		rollstore[DTStationIndex(0,dtWheel,dtSector,dtStation)];
	      //Loop over all the rolls asociated to RB4
	      for (std::set<RPCDetId>::iterator iteraRoll=rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
		const RPCRoll* rollasociated = 
		  dynamic_cast<const RPCRoll*> (rpcGeo->idToDetUnit(*iteraRoll));
		//To get the roll's surface
		const BoundPlane & RPCSurfaceRB4 = rollasociated->surface(); 
		
		const GeomDet* gdet=dtGeo->idToDet(segMB3->geographicalId());
		const BoundPlane & DTSurfaceMB3 = gdet->surface();


		std::cout<<"\t \t \t RollID: should be RB4"<<rollasociated->id()<<std::endl;
		std::cout<<"\t \t \t Making the extrapolation"<<std::endl;
		std::cout<<"\t \t \t DT Segment Direction in MB3 DTLocal "<<segMB3->localDirection()<<std::endl;
		std::cout<<"\t \t \t DT Segment Point in MB3 DTLocal "<<segMB3->localPosition()<<std::endl;
		
		GlobalPoint CenterPointRollGlobal=RPCSurfaceRB4.toGlobal(LocalPoint(0,0,0));
		std::cout<<"\t \t \t Center (0,0,0) of the RB4 Roll in Global"<<CenterPointRollGlobal<<std::endl;
		  
		LocalPoint CenterRollinDTFrame = DTSurfaceMB3.toLocal(CenterPointRollGlobal);
		std::cout<<"\t \t \t Center (0,0,0) Roll In DT MB3 Local"<<CenterRollinDTFrame<<std::endl;
		
		float D=CenterRollinDTFrame.z();
		std::cout<<"\t \t \t D="<<D<<"cm"<<std::endl;
		
		X=Xo3+dx3*D/dz3;
		Y=Yo3+dy3*D/dz3;
		Z=D;
		
		const RectangularStripTopology* top_=dynamic_cast<const RectangularStripTopology*>(&(rollasociated->topology()));
		LocalPoint xmin = top_->localPosition(0.);
		LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
		float rsize = fabs( xmax.x()-xmin.x() )*0.5;
		float stripl = top_->stripLength();
		
		std::cout<<"\t \t \t X Predicted in DT MB3 Local= "<<X<<"cm"<<std::endl;
		std::cout<<"\t \t \t Y Predicted in DT MB3 Local= "<<Y<<"cm"<<std::endl;
		std::cout<<"\t \t \t Z Predicted in DT MB3 Local= "<<Z<<"cm"<<std::endl;
		
		GlobalPoint GlobalPointExtrapolated = DTSurfaceMB3.toGlobal(LocalPoint(X,Y,Z));
		std::cout<<"\t \t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
		
		LocalPoint PointExtrapolatedRPCFrame = RPCSurfaceRB4.toLocal(GlobalPointExtrapolated);
		std::cout<<"\t \t \t Point Extrapolated in RPC RB4 Local"<<PointExtrapolatedRPCFrame<< std::endl;
		
		std::cout<<"\t \t \t Does the extrapolation go inside this roll?"<<std::endl;
		//conditions to find the right roll to extrapolate
		if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01  &&
		   fabs(PointExtrapolatedRPCFrame.x()) < rsize &&
		   fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
		  
		  std::cout<<"\t \t \t \t yes"<<std::endl;
		  //getting the number of the strip
		  const float stripPredicted=
		    rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		  
		  std::cout<<"\t \t \t \t Candidate"<<rollasociated->id()<<" "<<"(from DT Segment) STRIP---> "<<stripPredicted<< std::endl;
		  
		  //--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
		  
		  RPCDetId  rollId = rollasociated->id();
		  uint32_t id = rollId.rawId();
		  
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
		  std::cout << "\t \t \t \t One for counterPREDICT"<<std::endl;
		  totalcounter[0]++;
		  buff=counter[0];
		  buff[rollId]++;
		  counter[0]=buff;		
		    
		  bool anycoincidence=false;
		  int stripDetected = 0;
		  RPCDigiCollection::Range rpcRangeDigi = 
		    rpcDigis->get(rollasociated->id());
		  		  
		  //loop over the digis in the event
		  for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
		    std::cout<<"\t \t \t \t \t Digi "<<*digiIt<<std::endl;//print the digis in the event
		    stripDetected=digiIt->strip();
		    if(fabs((float)(stripDetected) - stripPredicted)<widestripsRB4){//Detected Vs Predicted
		      std::cout <<"\t \t \t \t \t COINCEDENCE Predict "<<stripPredicted<<" Detect "<<stripDetected<<std::endl;
		      anycoincidence=true;
		    }
		  }
		  if (anycoincidence){
		    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(stripPredicted);
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
		    std::cout <<"\t \t \t \t \t XXXXX THIS PREDICTION DOESN'T HAVE ANY CORRESPONDENCE WITH THE DATA"<<std::endl;
		    std::cout << "\t \t \t \t \t One for counterFAIL"<<std::endl;
		    ofrej<<"Wh "<<dtWheel<<" | St "<<dtStation
			 <<"  | Se "<<dtSector<<" | Event "
			 <<iEvent.id().event()<<std::endl;
		    
		  }
		}
		else{
		  std::cout<<"\t \t \t \t no"<<std::endl;
		}//Condition for the right match
	      }//loop over all the rolls FOR RB3 --------------------------------------------------------
	      correspondenceinRB3=true; 
	    }
	  }
	  if(!correspondenceinRB3){
	    std::cout<<"\t \t \t \t no corresp in RB3 or wrong direction"
		     <<std::endl;
	  }
	}
	else{
	  std::cout<<"\t \t Strange Segment"<<std::endl;
	}//Is not a 4D Segment neither a 2D in MB4
      }
      else {
	std::cout<<"\t \t no"<<std::endl;
      }//There is one segment in the chamber?
    }//loop over the segments
  }else {
    std::cout<<"This Event doesn't have any DT4DSegment"<<std::endl;
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
    std::cout <<id<<" ef = "<<ef*100.<<"+-"<<er*100.<<"%"<<std::endl;
  }
  float tote = float(totalcounter[1])/float(totalcounter[0]);
  float totr = sqrt(tote*(1.-tote)/float(totalcounter[0]));
  std::cout <<"\n\n"<<"Total efficiency "<<tote*100 <<" +- "<<totr*100
	    <<" %"<<std::endl;

  
  std::vector<uint32_t>::iterator meIt;
  for(meIt = _idList.begin(); meIt != _idList.end(); ++meIt){

    char detUnitLabel[128];
    char meIdRPC [128];
    char meIdDT [128];
    char effIdRPC [128];
    sprintf(detUnitLabel ,"%d",*meIt);
    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
    sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(effIdRPC,"EfficienyFromDTExtrapolation_%s",detUnitLabel);
    
    std::map<std::string, MonitorElement*> meMap=meCollection[*meIt];

    for(unsigned int i=1;i<=100;++i){
      
      if(meMap[meIdDT]->getBinContent(i) != 0){
	float eff = meMap[meIdRPC]->getBinContent(i)/meMap[meIdDT]->getBinContent(i);
	float erreff = sqrt(eff*(1-eff)/meMap[meIdDT]->getBinContent(i));
	meMap[effIdRPC]->setBinContent(i,eff);
	meMap[effIdRPC]->setBinError(i,erreff);
	
      }
    }
  }
  
  if(EffSaveRootFile) dbe->save(EffRootFileName);
  //  theFile->Write();
  //  theile->Close();
  std::cout<<"End Job"<<std::endl;
}

RPCMonitorEfficiency::~RPCMonitorEfficiency(){}
