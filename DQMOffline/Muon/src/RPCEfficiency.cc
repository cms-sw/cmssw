/***************************************
Author: 
Camilo Carrillo
Universidad de los Andes Bogota Colombia
camilo.carrilloATcern.ch
****************************************/


#include "DQMOffline/Muon/interface/RPCEfficiency.h"

// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/MakerMacros.h"

#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>

#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>

#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>

#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <cmath>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TString.h"


void RPCEfficiency::beginJob(const edm::EventSetup& iSetup){
  std::cout<<"Begin beginJob"<<std::endl;
  std::cout <<"\t Getting the RPC Geometry"<<std::endl;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  std::cout <<"\t Getting the DT Geometry"<<std::endl;
  iSetup.get<MuonGeometryRecord>().get(dtGeo);

  //std::cout <<"\t Getting the CSC Geometry"<<std::endl;
  //iSetup.get<MuonGeometryRecord>().get(cscGeo);

  
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	
	if(rpcId.region()==0)allrollstoreBarrel.insert(rpcId);
	
	int region=rpcId.region();
	
	if(region==0&&(incldt||incldtMB4)){
	  //std::cout<<"--Filling the dtstore"<<rpcId<<std::endl;
	  int wheel=rpcId.ring();
	  int sector=rpcId.sector();
	  int station=rpcId.station();
	  DTStationIndex ind(region,wheel,sector,station);
	  std::set<RPCDetId> myrolls;
	  if (rollstoreDT.find(ind)!=rollstoreDT.end()) myrolls=rollstoreDT[ind];
	  myrolls.insert(rpcId);
	  rollstoreDT[ind]=myrolls;
	}
	else if(inclcsc){
	  //std::cout<<"--Filling the cscstore"<<rpcId<<std::endl;
	  int region=rpcId.region();
          int station=rpcId.station();
          int ring=rpcId.ring();
          int cscring=ring;
          int cscstation=station;
	  RPCGeomServ rpcsrv(rpcId);
	  int rpcsegment = rpcsrv.segment();
	  int cscchamber = rpcsegment;
          if((station==2||station==3)&&ring==3){//Adding Ring 3 of RPC to the CSC Ring 2
            cscring = 2;
          }
	  if((station==4)&&(ring==2||ring==3)){//RE4 have just ring 1
            cscstation=3;
            cscring=2;
          }
          CSCStationIndex ind(region,cscstation,cscring,cscchamber);
          std::set<RPCDetId> myrolls;
	  if (rollstoreCSC.find(ind)!=rollstoreCSC.end()){
            myrolls=rollstoreCSC[ind];
          }
          
          myrolls.insert(rpcId);
          rollstoreCSC[ind]=myrolls;
        }
      }
    }
  }
  //std::cout<<"Containers Ready"<<std::endl;
}


RPCEfficiency::RPCEfficiency(const edm::ParameterSet& iConfig){
  //std::cout<<"Begin Constructor"<<std::endl;
  
  std::map<RPCDetId, int> buff;
  counter.clear();
  counter.reserve(3);
  counter.push_back(buff);
  counter.push_back(buff);
  counter.push_back(buff);
  totalcounter.clear();
  totalcounter.reserve(3);
  totalcounter[0]=0;
  totalcounter[1]=0;
  totalcounter[2]=0;

  incldt=iConfig.getUntrackedParameter<bool>("incldt",true);
  incldtMB4=iConfig.getUntrackedParameter<bool>("incldtMB4",true);
  inclcsc=iConfig.getUntrackedParameter<bool>("inclcsc",true);
  MinimalResidual= iConfig.getUntrackedParameter<double>("MinimalResidual",2.);
  MinimalResidualRB4=iConfig.getUntrackedParameter<double>("MinimalResidualRB4",4.);
  MinCosAng=iConfig.getUntrackedParameter<double>("MinCosAng",0.9999);
  MaxD=iConfig.getUntrackedParameter<double>("MaxD",20.);
  MaxDrb4=iConfig.getUntrackedParameter<double>("MaxDrb4",30.);
  MaxStripToCountInAverage=iConfig.getUntrackedParameter<double>("MaxStripToCountInAverage",5.);
  MaxStripToCountInAverageRB4=iConfig.getUntrackedParameter<double>("MaxStripToCountInAverageRB4",7.);
  muonRPCDigis=iConfig.getUntrackedParameter<std::string>("muonRPCDigis","muonRPCDigis");
  cscSegments=iConfig.getUntrackedParameter<std::string>("cscSegments","cscSegments");
  dt4DSegments=iConfig.getUntrackedParameter<std::string>("dt4DSegments","dt4DSegments");

  // Giuseppe
  nameInLog = iConfig.getUntrackedParameter<std::string>("moduleLogName", "RPC_Eff");
  EffSaveRootFile  = iConfig.getUntrackedParameter<bool>("EffSaveRootFile", false); 
  EffSaveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EffEventsInterval", 10000); 
  EffRootFileName  = iConfig.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiencyFirst.root"); 
  //Interface
  dbe = edm::Service<DQMStore>().operator->();
  _idList.clear(); 
  
  std::string folder = "RPC/MuonSegEff/";
  dbe->setCurrentFolder(folder);
  
  statistics = dbe->book1D("Statistics","All Statistics",33,0.5,33.5);
  statistics->setBinLabel(1,"Events ",1);
  statistics->setBinLabel(2,"Events with DT segments",1);
  statistics->setBinLabel(3,"Events with 1 DT segment",1);
  statistics->setBinLabel(4,"Events with 2 DT segments",1);
  statistics->setBinLabel(5,"Events with 3 DT segments",1);
  statistics->setBinLabel(6,"Events with 4 DT segments",1);
  statistics->setBinLabel(7,"Events with 5 DT segments",1);
  statistics->setBinLabel(8,"Events with 6 DT segments",1);
  statistics->setBinLabel(9,"Events with 7 DT segments",1);
  statistics->setBinLabel(10,"Events with 8 DT segments",1);
  statistics->setBinLabel(11,"Events with 9 DT segments",1);
  statistics->setBinLabel(12,"Events with 10 DT segments",1);
  statistics->setBinLabel(13,"Events with 11 DT segments",1);
  statistics->setBinLabel(14,"Events with 12 DT segments",1);
  statistics->setBinLabel(15,"Events with 13 DT segments",1);
  statistics->setBinLabel(16,"Events with 14 DT segments",1);
  statistics->setBinLabel(17,"Events with 15 DT segments",1);
  statistics->setBinLabel(18,"Events with CSC segments",1);
  statistics->setBinLabel(16+3,"Events with 1 CSC segment",1);
  statistics->setBinLabel(16+4,"Events with 2 CSC segments",1);
  statistics->setBinLabel(16+5,"Events with 3 CSC segments",1);
  statistics->setBinLabel(16+6,"Events with 4 CSC segments",1);
  statistics->setBinLabel(16+7,"Events with 5 CSC segments",1);
  statistics->setBinLabel(16+8,"Events with 6 CSC segments",1);
  statistics->setBinLabel(16+9,"Events with 7 CSC segments",1);
  statistics->setBinLabel(16+10,"Events with 8 CSC segments",1);
  statistics->setBinLabel(16+11,"Events with 9 CSC segments",1);
  statistics->setBinLabel(16+12,"Events with 10 CSC segments",1);
  statistics->setBinLabel(16+13,"Events with 11 CSC segments",1);
  statistics->setBinLabel(16+14,"Events with 12 CSC segments",1);
  statistics->setBinLabel(16+15,"Events with 13 CSC segments",1);
  statistics->setBinLabel(16+16,"Events with 14 CSC segments",1);
  statistics->setBinLabel(16+17,"Events with 15 CSC segments",1);

}

RPCEfficiency::~RPCEfficiency()
{
  //std::cout<<"Begin Destructor "<<std::endl;
}

void RPCEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  statistics->Fill(1);

  using namespace edm;
  
  std::map<RPCDetId, int> buff;

  char layerLabel[128];
  char meIdRPC [128];
  char meIdDT [128];
  //char meIdCSC [128];

  //std::cout <<"\t Getting the RPC Digis"<<std::endl;
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel(muonRPCDigis, rpcDigis);

  if(incldt){

    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    iEvent.getByLabel(dt4DSegments, all4DSegments);

    if(all4DSegments->size()>0){
      statistics->Fill(2);
  
      std::map<DTChamberId,int> scounter;
      DTRecSegment4DCollection::const_iterator segment;  
  
      for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
	scounter[segment->chamberId()]++;
      }    
  
      statistics->Fill(all4DSegments->size()+2);

      for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
    
	DTChamberId DTId = segment->chamberId();
    
    
	if(scounter[DTId]==1 && DTId.station()!=4){	
 
	  int dtWheel = DTId.wheel();
	  int dtStation = DTId.station();
	  int dtSector = DTId.sector();      

	  LocalPoint segmentPosition= segment->localPosition();
	  LocalVector segmentDirection=segment->localDirection();
      
	  const GeomDet* gdet=dtGeo->idToDet(segment->geographicalId());
	  const BoundPlane & DTSurface = gdet->surface();
      
	  //check if the dimension of the segment is 4 
      
	  if(segment->dimension()==4){
	
	    float Xo=segmentPosition.x();
	    float Yo=segmentPosition.y();
	    float Zo=segmentPosition.z();
	    float dx=segmentDirection.x();
	    float dy=segmentDirection.y();
	    float dz=segmentDirection.z();

	    std::set<RPCDetId> rollsForThisDT = rollstoreDT[DTStationIndex(0,dtWheel,dtSector,dtStation)];
       
	    assert(rollsForThisDT.size()>=1);

	    //Loop over all the rolls
	    for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
	      const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
	      RPCDetId rpcId = rollasociated->id();
	      const BoundPlane & RPCSurface = rollasociated->surface(); 
	  

	      GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
	  
	      LocalPoint CenterRollinDTFrame = DTSurface.toLocal(CenterPointRollGlobal);
	    
	      float D=CenterRollinDTFrame.z();
	  
	      float X=Xo+dx*D/dz;
	      float Y=Yo+dy*D/dz;
	      float Z=D;
	
	      const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&(rollasociated->topology()));
	      LocalPoint xmin = top_->localPosition(0.);
	      LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
	      float rsize = fabs( xmax.x()-xmin.x() );
	      float stripl = top_->stripLength();
	      //float stripw = top_->pitch();
	  	  
	  
	      float extrapolatedDistance = sqrt((X-Xo)*(X-Xo)+(Y-Yo)*(Y-Yo)+(Z-Zo)*(Z-Zo));
	      if(extrapolatedDistance<=MaxD){ 
	    
		GlobalPoint GlobalPointExtrapolated = DTSurface.toGlobal(LocalPoint(X,Y,Z));
	    
		LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);
	    

		if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01 && 
		   fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 && 
		   fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){
	      
		  RPCDetId  rollId = rollasociated->id();

		  const float stripPredicted = 
		    rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		
		
	      		
		  //--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
		
		  RPCGeomServ rpcsrv(rollId);
		  std::string nameRoll = rpcsrv.name();
		  //if(_idList.at(nameRoll)==null) 
		  bool deja=false;
		  std::vector<std::string>::iterator meIt;
		  for(meIt = _idList.begin(); meIt != _idList.end(); ++meIt){
		    if(*meIt==nameRoll){ 
		      deja=true;
		      break;
		    }
		  }
		  if(!deja){
		    _idList.push_back(nameRoll);
		  }
		
		  char detUnitLabel[128];
		  sprintf(detUnitLabel ,"%s",nameRoll.c_str());
		  sprintf(layerLabel ,"%s",nameRoll.c_str());
		
		  std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);
				
		  if (meItr == meCollection.end()){
		    meCollection[nameRoll] = bookDetUnitSeg(rollId,rollasociated->nstrips());
		  }
		
		  std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
		
		  sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
		  meMap[meIdDT]->Fill(stripPredicted);
		
		  //-----------------------------------------------------
		
		  totalcounter[0]++;
		  buff=counter[0];
		  buff[rollId]++;
		  counter[0]=buff;
		
		  bool anycoincidence=false;
		  double sumStripDetected = 0.;  

		  int stripDetected = 0;
		  int stripCounter = 0;
		  RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());


		  for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
		    stripDetected=digiIt->strip(); 
		    if(fabs((float)stripDetected-stripPredicted)<MaxStripToCountInAverage){
		      sumStripDetected=sumStripDetected+stripDetected;
		      stripCounter++;
		    }
		
		    sprintf(meIdRPC,"BXDistribution_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(digiIt->bx());
		
		    sprintf(meIdRPC,"RealDetectedOccupancyFromDT_%s",detUnitLabel);
		    meMap[meIdRPC]->Fill(stripDetected); //have a look to this!
		  }
	      
	      
		  if(stripCounter!=0){
		    double meanStripDetected = sumStripDetected/((double)stripCounter);
		
	      
		    LocalPoint meanstripDetectedLocalPoint = top_->localPosition((float)(meanStripDetected)-0.5);
	      
		    float meanrescms = PointExtrapolatedRPCFrame.x()-meanstripDetectedLocalPoint.x();          
			      

		    if(fabs(meanrescms) < MinimalResidual ){
		
		
		      anycoincidence=true;
		      totalcounter[1]++;
		      buff=counter[1];
		      buff[rollId]++;
		      counter[1]=buff;
		  
		      sprintf(meIdRPC,"RPCDataOccupancyFromDT_%s",detUnitLabel);
		      meMap[meIdRPC]->Fill(stripPredicted);
		    }
		  }else{
		  }
	      
		  if(anycoincidence==false) {
		    totalcounter[2]++;
		    buff=counter[2];
		    buff[rollId]++;
		    counter[2]=buff;		
		  
		    ofrej<<"DTs \t Wh "<<dtWheel
			 <<"\t St "<<dtStation
			 <<"\t Se "<<dtSector
			 <<"\t Roll "<<rollasociated->id()
			 <<"\t Event "
			 <<iEvent.id().event()
			 <<"\t Run "	
			 <<iEvent.id().run()	
			 <<std::endl;
		  }
		}else {
		}//Condition for the right match
	      }else{
	      }//D so big
	    }//loop over all the rolls asociated
	  }//Is the segment 4D?
	}else {
	}
      }
    }
    else {
    }
  }
  
  if(incldtMB4){

    //#include "RPCEfficiencyrb4part.inl"

    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    iEvent.getByLabel(dt4DSegments, all4DSegments);
    
    if(all4DSegments->size()>0){
  
      std::map<DTChamberId,int> scounter;
      DTRecSegment4DCollection::const_iterator segment;  
  
      for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
	scounter[segment->chamberId()]++;
      }    
  
      for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
    
	DTChamberId DTId = segment->chamberId();
    
    
	if(scounter[DTId] == 1 && DTId.station()==4){
	  int dtWheel = DTId.wheel();
	  int dtStation = DTId.station();
	  int dtSector = DTId.sector();
      
	  LocalPoint segmentPosition= segment->localPosition();
	  LocalVector segmentDirection=segment->localDirection();
            
	  //check if the dimension of the segment is 2 and the station is 4
            

	  if(segment->dimension()==2){
	  
	    LocalVector segmentDirectionMB4=segmentDirection;
	    LocalPoint segmentPositionMB4=segmentPosition;
	
	    bool compatiblesegments=false;
	    float dx=segmentDirectionMB4.x();
	    float dz=segmentDirectionMB4.z();
	
	    const BoundPlane& DTSurface4 = dtGeo->idToDet(DTId)->surface();
	
	
	    DTRecSegment4DCollection::const_iterator segMB3;  
	
	    for(segMB3=all4DSegments->begin();segMB3!=all4DSegments->end();++segMB3){

	      DTChamberId dtid3 = segMB3->chamberId();  
	      if(dtid3.station()==3&&dtid3.wheel()==DTId.wheel()&&scounter[dtid3] == 1&&segMB3->dimension()==4){

		const GeomDet* gdet3=dtGeo->idToDet(segMB3->geographicalId());
		const BoundPlane & DTSurface3 = gdet3->surface();
	      
		float dx3=segMB3->localDirection().x();
		float dy3=segMB3->localDirection().y();
		float dz3=segMB3->localDirection().z();
	    
		LocalVector segDirMB4inMB3Frame=DTSurface3.toLocal(DTSurface4.toGlobal(segmentDirectionMB4));
	    
		double cosAng=fabs(dx*dx3+dz*dz3/sqrt((dx3*dx3+dz3*dz3)*(dx*dx+dz*dz)));
		assert(fabs(cosAng)<=1.);
	    
		if(cosAng>MinCosAng){
		  compatiblesegments=true;
		  if(dtSector==13){
		    dtSector=4;
		  }
		  if(dtSector==14){
		    dtSector=10;
		  }
	    
		  std::set<RPCDetId> rollsForThisDT = rollstoreDT[DTStationIndex(0,dtWheel,dtSector,4)]; //It should be always 4
	      
		  assert(rollsForThisDT.size()>=1);
     	      
		  for (std::set<RPCDetId>::iterator iteraRoll=rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
		    const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll); //roll asociado a MB4
		    const BoundPlane & RPCSurfaceRB4 = rollasociated->surface(); //surface MB4
		    const GeomDet* gdet=dtGeo->idToDet(segMB3->geographicalId()); 
		    const BoundPlane & DTSurfaceMB3 = gdet->surface(); // surface MB3
		
		    GlobalPoint CenterPointRollGlobal=RPCSurfaceRB4.toGlobal(LocalPoint(0,0,0));
		
		    LocalPoint CenterRollinMB3Frame = DTSurfaceMB3.toLocal(CenterPointRollGlobal);

		    float D=CenterRollinMB3Frame.z();
		
		    float Xo3=segMB3->localPosition().x();
		    float Yo3=segMB3->localPosition().y();
		    float Zo3=segMB3->localPosition().z();

		    float X=Xo3+dx3*D/dz3;
		    float Y=Yo3+dy3*D/dz3;
		    float Z=D;

		
		    const RectangularStripTopology* top_
		      =dynamic_cast<const RectangularStripTopology*>(&(rollasociated->topology())); //Topology roll asociated MB4
		    LocalPoint xmin = top_->localPosition(0.);
		    LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
		    float rsize = fabs( xmax.x()-xmin.x() );
		    float stripl = top_->stripLength();
		    //float stripw = top_->pitch();

		

		    float extrapolatedDistance = sqrt((X-Xo3)*(X-Xo3)+(Y-Yo3)*(Y-Yo3)+(Z-Zo3)*(Z-Zo3));

		    if(extrapolatedDistance<=MaxDrb4){ 
		  
		      GlobalPoint GlobalPointExtrapolated = DTSurfaceMB3.toGlobal(LocalPoint(X,Y,Z));

		      LocalPoint PointExtrapolatedRPCFrame = RPCSurfaceRB4.toLocal(GlobalPointExtrapolated);
	    
		
		      if(fabs(PointExtrapolatedRPCFrame.z()) < 0.01  &&
			 fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 &&
			 fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
		    
			RPCDetId  rollId = rollasociated->id();
		    
			const float stripPredicted=
			  rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		  
		    
		    

			//--------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
		  
			RPCGeomServ rpcsrv(rollId);
			std::string nameRoll = rpcsrv.name();
		  
			bool deja=false;
			std::vector<std::string>::iterator meIt;
			for(meIt = _idList.begin(); meIt != _idList.end(); ++meIt){
			  if(*meIt==nameRoll){ 
			    deja=true;
			    break;
			  }
			}
			if(!deja){
			  _idList.push_back(nameRoll);
			}
		    
			char detUnitLabel[128];
			sprintf(detUnitLabel ,"%s",nameRoll.c_str());
			sprintf(layerLabel ,"%s",nameRoll.c_str());
		    
		    
			std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);
			if (meItr == meCollection.end()){
			  meCollection[nameRoll] = bookDetUnitSeg(rollId,rollasociated->nstrips());
			}
		    
			std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
		    
			sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
			meMap[meIdDT]->Fill(stripPredicted);
		    
		    
			//-------------------------------------------------
		    
			totalcounter[0]++;
			buff=counter[0];
			buff[rollId]++;
			counter[0]=buff;		
		    
			bool anycoincidence=false;
			double sumStripDetected = 0.;  
		    
			int stripDetected = 0;
			int stripCounter = 0;
			RPCDigiCollection::Range rpcRangeDigi = rpcDigis->get(rollasociated->id());
		    
		    
			for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
			  stripDetected=digiIt->strip(); 
			  if(fabs((float)stripDetected-stripPredicted)<MaxStripToCountInAverageRB4){
			    sumStripDetected=sumStripDetected+stripDetected;
			    stripCounter++;
			  }
		      
			  sprintf(meIdRPC,"BXDistribution_%s",detUnitLabel);
			  meMap[meIdRPC]->Fill(digiIt->bx());

			  sprintf(meIdRPC,"RealDetectedOccupancyFromDT_%s",detUnitLabel);
			  meMap[meIdRPC]->Fill(stripDetected);
		      
			}
		    
		    
			if(stripCounter!=0){
			  double meanStripDetected = meanStripDetected=sumStripDetected/((double)stripCounter);
		      
		      
			  LocalPoint meanstripDetectedLocalPoint = top_->localPosition((float)(meanStripDetected)-0.5);
		      
			  float meanrescms = PointExtrapolatedRPCFrame.x()-meanstripDetectedLocalPoint.x();          
		      
		      
			  if(fabs(meanrescms) < MinimalResidualRB4 ){
			
			
			    anycoincidence=true;
			    totalcounter[1]++;
			    buff=counter[1];
			    buff[rollId]++;
			    counter[1]=buff;		
			
			    sprintf(meIdRPC,"RPCDataOccupancyFromDT_%s",detUnitLabel);
			    meMap[meIdRPC]->Fill(stripPredicted);
			
			  }
			}else{
			}
			if(anycoincidence==false){
			  totalcounter[2]++;
			  buff=counter[2];
			  buff[rollId]++;
			  counter[2]=buff;		
			  ofrej<<"MB4 \t Wh "<<dtWheel
			       <<"\t St "<<dtStation
			       <<"\t Se "<<dtSector
			       <<"\t Roll "<<rollasociated->id()
			       <<"\t Event "
			       <<iEvent.id().event()
			       <<"\t Run "
			       <<iEvent.id().run()
			       <<std::endl;
			}
		      }else{
		      }
		    }else{
		    }
		  }//loop over all the rollsasociated
		}else{
		  compatiblesegments=false;
		}
	      }else{//if dtid3.station()==3&&dtid3.sector()==DTId.sector()&&dtid3.wheel()==DTId.wheel()&&segMB3->dim()==4
	      }
	    }//lood over all the segments looking for one in MB3 
	  }else{
	  }
	}else{
	}//De aca para abajo esta en dtpart.inl
      }
    }else{
    }
  }
  
  if(inclcsc){
    //#include "RPCEfficiencycscpart.inl"
    
    edm::Handle<CSCSegmentCollection> allCSCSegments;
    iEvent.getByLabel(cscSegments, allCSCSegments);
    
    if(allCSCSegments->size()>0){
      statistics->Fill(18);
      
      std::map<CSCDetId,int> CSCSegmentsCounter;
      CSCSegmentCollection::const_iterator segment;
      
      int segmentsInThisEventInTheEndcap=0;
      
      for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
	CSCSegmentsCounter[segment->cscDetId()]++;
	segmentsInThisEventInTheEndcap++;
      }    
     
      statistics->Fill(allCSCSegments->size()+18);
      
    }
  } 
}

void RPCEfficiency::endRun(const edm::Run& r, const edm::EventSetup& iSetup){
  if (EffSaveRootFile){
    dbe->save(EffRootFileName);
  }
}


void RPCEfficiency::endJob()
{
  //std::cout<<"Begin End Job"<<std::endl;
  dbe =0;
}
