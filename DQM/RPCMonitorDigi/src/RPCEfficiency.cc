/***************************************
Original Author: Camilo Carrillo
****************************************/

#include "DQM/RPCMonitorDigi/interface/RPCEfficiency.h"
#include <sstream>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

//FW Core
#include "FWCore/MessageLogger/interface/MessageLogger.h"

int distsector_tmp(int sector1,int sector2){
 
  if(sector1==13) sector1=4;
  if(sector1==14) sector1=10;
   
  if(sector2==13) sector2=4;
  if(sector2==14) sector2=10;
   
  int distance = abs(sector1 - sector2);
  if(distance>6) distance = 12-distance;
  return distance;
}


RPCEfficiency::RPCEfficiency(const edm::ParameterSet& iConfig){
  incldt=iConfig.getUntrackedParameter<bool>("incldt",true);
  incldtMB4=iConfig.getUntrackedParameter<bool>("incldtMB4",true);
  inclcsc=iConfig.getUntrackedParameter<bool>("inclcsc",true);
  debug=iConfig.getUntrackedParameter<bool>("debug",false);
  
  rangestrips = iConfig.getUntrackedParameter<double>("rangestrips",4.);
  rangestripsRB4=iConfig.getUntrackedParameter<double>("rangestripsRB4",4.);
  dupli = iConfig.getUntrackedParameter<int>("DuplicationCorrection",2); 
  MinCosAng=iConfig.getUntrackedParameter<double>("MinCosAng",0.96);
  MaxD=iConfig.getUntrackedParameter<double>("MaxD",80.);
  MaxDrb4=iConfig.getUntrackedParameter<double>("MaxDrb4",150.);


  cscSegments= consumes<CSCSegmentCollection>(iConfig.getParameter<edm::InputTag>("cscSegments"));
  dt4DSegments =  consumes<DTRecSegment4DCollection>(iConfig.getParameter<edm::InputTag>("dt4DSegments"));
  RPCRecHitLabel_ =  consumes<RPCRecHitCollection>(iConfig.getParameter<edm::InputTag>("RecHitLabel"));

  folderPath=iConfig.getUntrackedParameter<std::string>("folderPath","RPC/RPCEfficiency/");
   
  EffSaveRootFile  = iConfig.getUntrackedParameter<bool>("EffSaveRootFile", false); 
  EffRootFileName  = iConfig.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiency.root"); 

}


void RPCEfficiency::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & run, edm::EventSetup const & iSetup ) {

  std::string folder;
  ibooker.cd();
  ibooker.setCurrentFolder(folderPath);
  statistics = ibooker.book1D("Statistics","All Statistics",33,0.5,33.5);

   
  statistics->setBinLabel(1,"Events ",1);
  statistics->setBinLabel(2,"Events with DT seg",1);
  
  std::stringstream sstr;
  for( int i = 1; i<=15; i++ ){ //DT form bin 3 to bin 17
    sstr.str("");
    sstr<<i<<" DT seg";
    statistics->setBinLabel(i+2, sstr.str() ,1);
  }

  statistics->setBinLabel(18,"Events with CSC seg",1);
  for( int i = 1; i<=15; i++ ){ //CSC form bin 19 to bin 33
    sstr.str("");
    sstr<<i<<" CSC seg";
    statistics->setBinLabel(i+18, sstr.str(),1);
  }
  

   LogDebug("rpcefficiency")<<"booking Global histograms with "<<folderPath;
   
  folder = folderPath+"MuonSegEff/"+"Residuals/Barrel";
  ibooker.setCurrentFolder(folder);
 
  //Barrel
  std::stringstream histoName, histoTitle;

  for (int layer = 1 ; layer<= 6 ;layer++){
    histoName.str("");
    histoTitle.str("");
    histoName<<"GlobalResidualsClu1La"<<layer;
    histoTitle<<"RPC Residuals Layer "<<layer<<" Cluster Size 1"; 
    hGlobalResClu1La[layer-1] = ibooker.book1D(histoName.str(), histoTitle.str(),101,-10.,10.);
 
    histoName.str("");
    histoTitle.str("");
    histoName<<"GlobalResidualsClu2La"<<layer;
    histoTitle<<"RPC Residuals Layer "<<layer<<" Cluster Size 2"; 
    hGlobalResClu2La[layer-1] = ibooker.book1D(histoName.str(), histoTitle.str(),101,-10.,10.);
    
    histoName.str("");
    histoTitle.str("");
    histoName<<"GlobalResidualsClu3La"<<layer;
    histoTitle<<"RPC Residuals Layer "<<layer<<" Cluster Size 3"; 
    hGlobalResClu3La[layer-1] = ibooker.book1D(histoName.str(), histoTitle.str(),101,-10.,10.);
    
  }
  
   LogDebug("rpcefficiency")<<"Booking Residuals for EndCap";
  folder = folderPath+"MuonSegEff/Residuals/EndCap";
  ibooker.setCurrentFolder(folder);

  //Endcap   
  hGlobalResClu1R3C = ibooker.book1D("GlobalResidualsClu1R3C","RPC Residuals Ring 3 Roll C Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R3B = ibooker.book1D("GlobalResidualsClu1R3B","RPC Residuals Ring 3 Roll B Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R3A = ibooker.book1D("GlobalResidualsClu1R3A","RPC Residuals Ring 3 Roll A Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R2C = ibooker.book1D("GlobalResidualsClu1R2C","RPC Residuals Ring 2 Roll C Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R2B = ibooker.book1D("GlobalResidualsClu1R2B","RPC Residuals Ring 2 Roll B Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R2A = ibooker.book1D("GlobalResidualsClu1R2A","RPC Residuals Ring 2 Roll A Cluster Size 1",101,-10.,10.);

  hGlobalResClu2R3C = ibooker.book1D("GlobalResidualsClu2R3C","RPC Residuals Ring 3 Roll C Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R3B = ibooker.book1D("GlobalResidualsClu2R3B","RPC Residuals Ring 3 Roll B Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R3A = ibooker.book1D("GlobalResidualsClu2R3A","RPC Residuals Ring 3 Roll A Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R2C = ibooker.book1D("GlobalResidualsClu2R2C","RPC Residuals Ring 2 Roll C Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R2B = ibooker.book1D("GlobalResidualsClu2R2B","RPC Residuals Ring 2 Roll B Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R2A = ibooker.book1D("GlobalResidualsClu2R2A","RPC Residuals Ring 2 Roll A Cluster Size 2",101,-10.,10.);

  hGlobalResClu3R3C = ibooker.book1D("GlobalResidualsClu3R3C","RPC Residuals Ring 3 Roll C Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R3B = ibooker.book1D("GlobalResidualsClu3R3B","RPC Residuals Ring 3 Roll B Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R3A = ibooker.book1D("GlobalResidualsClu3R3A","RPC Residuals Ring 3 Roll A Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R2C = ibooker.book1D("GlobalResidualsClu3R2C","RPC Residuals Ring 2 Roll C Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R2B = ibooker.book1D("GlobalResidualsClu3R2B","RPC Residuals Ring 2 Roll B Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R2A = ibooker.book1D("GlobalResidualsClu3R2A","RPC Residuals Ring 2 Roll A Cluster Size 3",101,-10.,10.);


  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	
	RPCDetId rpcId = (*r)->id();
	int region=rpcId.region();
	
	LogDebug("rpcefficiency")<<"Booking for "<<rpcId.rawId();
	
	bookDetUnitSeg(ibooker, rpcId,(*r)->nstrips(),folderPath+"MuonSegEff/", meCollection[rpcId.rawId()] );
	
	if(region==0&&(incldt||incldtMB4)){
	  //LogDebug("rpcefficiency")<<"--Filling the dtstore"<<rpcId;
	  int wheel=rpcId.ring();
	  int sector=rpcId.sector();
	  int station=rpcId.station();
	  DTStationIndex ind(region,wheel,sector,station);
	  std::set<RPCDetId> myrolls;
	  if (rollstoreDT.find(ind)!=rollstoreDT.end()) myrolls=rollstoreDT[ind];
	  myrolls.insert(rpcId);
	  rollstoreDT[ind]=myrolls;
 
	}else if(region!=0 && inclcsc){
	  int station=rpcId.station();
          int ring=rpcId.ring();
          int cscring=ring;
          int cscstation=station;
	  RPCGeomServ rpcsrv(rpcId);
	  int rpcsegment = rpcsrv.segment();
	  int cscchamber = rpcsegment;

          if((station==2||station==3||station==4)&&ring==3){
            cscring = 2;
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

  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
       
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	 
	int region=rpcId.region();
	 
	if(region!=0 && inclcsc && (rpcId.ring()==2 || rpcId.ring()==3)){
	  int region=rpcId.region();                                                                                         
          int station=rpcId.station();                                                                                       
          int ring=rpcId.ring();                                                                                             
	  int cscring = ring;
	     
	  if((station==2||station==3||station==4)&&ring==3) cscring = 2; //CSC Ring 2 covers rpc ring 2 & 3                              

 
          int cscstation=station;                                                                                            
          RPCGeomServ rpcsrv(rpcId);                                                                                         
          int rpcsegment = rpcsrv.segment();                                                                                 
                                                                                                                             
	  int cscchamber = rpcsegment+1;                                                                                     
          if(cscchamber==37)cscchamber=1;                                                                                    
          CSCStationIndex ind(region,cscstation,cscring,cscchamber);                                                         
	  std::set<RPCDetId> myrolls;                                                                                        
          if (rollstoreCSC.find(ind)!=rollstoreCSC.end())myrolls=rollstoreCSC[ind];                                          
          myrolls.insert(rpcId);                                                                                             
          rollstoreCSC[ind]=myrolls;                                                                                         
                                                                                                                              
          cscchamber = rpcsegment-1;                                                                                         
          if(cscchamber==0)cscchamber=36;                                                                                    
          CSCStationIndex indDos(region,cscstation,cscring,cscchamber);                                                      
	  std::set<RPCDetId> myrollsDos;                                                                                     
          if (rollstoreCSC.find(indDos)!=rollstoreCSC.end()) myrollsDos=rollstoreCSC[indDos];                                 
          myrollsDos.insert(rpcId);                                                                                          
          rollstoreCSC[indDos]=myrollsDos;                                                                                                                                 
        }
      }
    }
  }
}//beginRun






RPCEfficiency::~RPCEfficiency(){}

void RPCEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
   

  edm::ESHandle<RPCGeometry> rpcGeo;
  edm::ESHandle<DTGeometry> dtGeo;  
  edm::ESHandle<CSCGeometry> cscGeo;
  
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  iSetup.get<MuonGeometryRecord>().get(dtGeo);
  iSetup.get<MuonGeometryRecord>().get(cscGeo);
  
  statistics->Fill(1);
  
  std::stringstream  meIdRPC, meIdDT, meIdCSC;
  
   LogDebug("rpcefficiency") <<"\t Getting the RPC RecHits";
  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByToken(RPCRecHitLabel_,rpcHits);  
  
  if(!rpcHits.isValid()) return;
  
  if(incldt){
     LogDebug("rpcefficiency")<<"\t Getting the DT Segments";
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    
    iEvent.getByToken(dt4DSegments, all4DSegments);
    
    if(all4DSegments.isValid()){
      
      if(all4DSegments->size()>0){
	
	if(all4DSegments->size()<=16) statistics->Fill(2);
	
	 LogDebug("rpcefficiency")<<"\t Number of DT Segments in this event = "<<all4DSegments->size();
	
	std::map<DTChamberId,int> DTSegmentCounter;
	DTRecSegment4DCollection::const_iterator segment;  
	
	for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
	  DTSegmentCounter[segment->chamberId()]++;
	}    
	
	statistics->Fill(all4DSegments->size()+2);
	  
	 LogDebug("rpcefficiency")<<"\t Loop over all the 4D Segments";
	for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
	  
	  DTChamberId DTId = segment->chamberId();
	  
	  
	  if(DTSegmentCounter[DTId]==1 && DTId.station()!=4){	
	    
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
	      
	       LogDebug("rpcefficiency")<<"DT  \t \t Loop over all the rolls asociated to this DT";
	      for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
		const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
		RPCDetId rpcId = rollasociated->id();
		const BoundPlane & RPCSurface = rollasociated->surface(); 
		
// 		RPCGeomServ rpcsrv(rpcId);
// 		std::string nameRoll = rpcsrv.name();
		
		GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
		
		LocalPoint CenterRollinDTFrame = DTSurface.toLocal(CenterPointRollGlobal);
		
		float D=CenterRollinDTFrame.z();
		
		float X=Xo+dx*D/dz;
		float Y=Yo+dy*D/dz;
		float Z=D;
		
		const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&(rollasociated->topology()));
		LocalPoint xmin = top_->localPosition(0.);
		 LogDebug("rpcefficiency")<<"DT  \t \t \t xmin of this  Roll "<<xmin<<"cm";
		LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
		 LogDebug("rpcefficiency")<<"DT  \t \t \t xmax of this  Roll "<<xmax<<"cm";
		float rsize = fabs( xmax.x()-xmin.x() );
		 LogDebug("rpcefficiency")<<"DT  \t \t \t Roll Size "<<rsize<<"cm";
		float stripl = top_->stripLength();
		float stripw = top_->pitch();
		
		float extrapolatedDistance = sqrt((X-Xo)*(X-Xo)+(Y-Yo)*(Y-Yo)+(Z-Zo)*(Z-Zo));
		
		if(extrapolatedDistance<=MaxD){ 
		  
		  GlobalPoint GlobalPointExtrapolated = DTSurface.toGlobal(LocalPoint(X,Y,Z));
		  LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);
		  
		  if(fabs(PointExtrapolatedRPCFrame.z()) < 10. && 
		     fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 && 
		     fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){
		    
		    RPCDetId  rollId = rollasociated->id();		      
		    RPCGeomServ rpcsrv(rollId);
		    std::string nameRoll = rpcsrv.name();
		     LogDebug("rpcefficiency")<<"DT  \t \t \t \t The RPCName is "<<nameRoll;		    
		    const float stripPredicted = 
		      rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		    
		     LogDebug("rpcefficiency")<<"DT  \t \t \t \t Candidate (from DT Segment) STRIP---> "<<stripPredicted<< std::endl;		  
		    //---- HISTOGRAM STRIP PREDICTED FROM DT ----
		    
		    std::map<std::string, MonitorElement*> meMap=meCollection[rpcId.rawId()];
		    meIdDT.str("");
		    meIdDT<<"ExpectedOccupancyFromDT_"<<rollId.rawId();
		    meMap[meIdDT.str()]->Fill(stripPredicted);
		    //-----------------------------------------------------
		      
		    
		    //-------RecHitPart Just For Residual--------
		    int countRecHits = 0;
		    int cluSize = 0;
		    float minres = 3000.;
		    
		    typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
		    rangeRecHits recHitCollection =  rpcHits->get(rollasociated->id());
		    RPCRecHitCollection::const_iterator recHit;
		      
		    for (recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++) {
		      countRecHits++;
		      
		   //    sprintf(meIdRPC,"BXDistribution_%d",rollasociated->id().rawId());
// 		      meMap[meIdRPC]->Fill(recHit->BunchX());
		      
		      LocalPoint recHitPos=recHit->localPosition();
		      float res=PointExtrapolatedRPCFrame.x()- recHitPos.x();	    
		       LogDebug("rpcefficiency")<<"DT  \t \t \t \t \t Found Rec Hit at "<<res<<"cm of the prediction.";
		      if(fabs(res)<fabs(minres)){
			minres=res;
			cluSize = recHit->clusterSize();
			 LogDebug("rpcefficiency")<<"DT  \t \t \t \t \t \t New Min Res "<<res<<"cm.";
		      }
		    }
		    
		    if(countRecHits==0){
		       LogDebug("rpcefficiency") <<"DT \t \t \t \t \t THIS ROLL DOESN'T HAVE ANY RECHIT";
		    }else{
		      assert(minres!=3000);     
		        
		      if(fabs(minres)<=(rangestrips+cluSize*0.5)*stripw){
			 LogDebug("rpcefficiency")<<"DT  \t \t \t \t \t \t True!";
			
			//	float cosal = dx/sqrt(dx*dx+dz*dz);    
			
			if(rollId.station()==1&&rollId.layer()==1)     { 
			  if(cluSize==1*dupli) {hGlobalResClu1La[0]->Fill(minres);}
			  else if(cluSize==2*dupli){ hGlobalResClu2La[0]->Fill(minres);} 
			  else if(cluSize==3*dupli){ hGlobalResClu3La[0]->Fill(minres);}}
			else if(rollId.station()==1&&rollId.layer()==2){ 
			  if(cluSize==1*dupli) {hGlobalResClu1La[1]->Fill(minres);}
			  else if(cluSize==2*dupli){ hGlobalResClu2La[1]->Fill(minres);} 
			  else if(cluSize==3*dupli){ hGlobalResClu3La[1]->Fill(minres);}}
			else if(rollId.station()==2&&rollId.layer()==1){ 
			  if(cluSize==1*dupli) {hGlobalResClu1La[2]->Fill(minres);}
			  else if(cluSize==2*dupli){ hGlobalResClu2La[2]->Fill(minres);} 
			  else if(cluSize==3*dupli){ hGlobalResClu3La[2]->Fill(minres);}
			}
			else if(rollId.station()==2&&rollId.layer()==2){ 
			  if(cluSize==1*dupli) {hGlobalResClu1La[3]->Fill(minres);}
			  if(cluSize==2*dupli){ hGlobalResClu2La[3]->Fill(minres);} 
			  else if(cluSize==3*dupli){ hGlobalResClu3La[3]->Fill(minres);}
			}
			else if(rollId.station()==3){ 
			  if(cluSize==1*dupli) {hGlobalResClu1La[4]->Fill(minres);}
			  else if(cluSize==2*dupli){ hGlobalResClu2La[4]->Fill(minres);} 
			  else if(cluSize==3*dupli){ hGlobalResClu3La[4]->Fill(minres);}
		      }
			meIdRPC.str("");
			meIdRPC<<"RPCDataOccupancyFromDT_"<<rollId.rawId();
			meMap[meIdRPC.str()]->Fill(stripPredicted);
		      }
		    }
		  }else{
		     LogDebug("rpcefficiency")<<"DT \t \t \t \t No the prediction is outside of this roll";
		  }//Condition for the right match
		}else{
		   LogDebug("rpcefficiency")<<"DT \t \t \t No, Exrtrapolation too long!, canceled";
		  }//D so big
	      }//loop over all the rolls asociated
	    }//Is the segment 4D?
	  }else {
	     LogDebug("rpcefficiency")<<"DT \t \t More than one segment in this chamber, or we are in Station 4";
	  }
	}
      } else {  
	 LogDebug("rpcefficiency")<<"DT This Event doesn't have any DT4DDSegment"; //is ther more than 1 segment in this event?
      }
    }
  }
  
  if(incldtMB4){
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    iEvent.getByToken(dt4DSegments, all4DSegments);
      
    if(all4DSegments.isValid() && all4DSegments->size()>0){

	std::map<DTChamberId,int> DTSegmentCounter;
	DTRecSegment4DCollection::const_iterator segment;  
        
	for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
	  DTSegmentCounter[segment->chamberId()]++;
	}    
	
	 LogDebug("rpcefficiency")<<"MB4 \t \t Loop Over all4DSegments";
	for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
	  
	  DTChamberId DTId = segment->chamberId();
	  	
	  if(DTSegmentCounter[DTId] == 1 && DTId.station()==4){
	    	  int dtWheel = DTId.wheel();
		  int dtStation = DTId.station();
		  int dtSector = DTId.sector();
		  
		  LocalPoint segmentPosition= segment->localPosition();
		  LocalVector segmentDirection=segment->localDirection();
		  
		  //check if the dimension of the segment is 2 and the station is 4

		  if(segment->dimension()==2){
		    LocalVector segmentDirectionMB4=segmentDirection;
		    LocalPoint segmentPositionMB4=segmentPosition;
		    
		    
		    const BoundPlane& DTSurface4 = dtGeo->idToDet(DTId)->surface();
		    
		    DTRecSegment4DCollection::const_iterator segMB3;  
		    
		    for(segMB3=all4DSegments->begin();segMB3!=all4DSegments->end();++segMB3){
		      
		      DTChamberId dtid3 = segMB3->chamberId();  
	        
		      if(distsector_tmp(dtid3.sector(),DTId.sector())<=1 
			 && dtid3.station()==3
			 && dtid3.wheel()==DTId.wheel()
			 && DTSegmentCounter[dtid3] == 1
			 && segMB3->dimension()==4){
			
			const GeomDet* gdet3=dtGeo->idToDet(segMB3->geographicalId());
			const BoundPlane & DTSurface3 = gdet3->surface();
			
			LocalVector segmentDirectionMB3 =  segMB3->localDirection();
			GlobalPoint segmentPositionMB3inGlobal = DTSurface3.toGlobal(segMB3->localPosition());
			
			GlobalVector segDirMB4inGlobalFrame=DTSurface4.toGlobal(segmentDirectionMB4);
			GlobalVector segDirMB3inGlobalFrame=DTSurface3.toGlobal(segmentDirectionMB3);
			
			float dx=segDirMB4inGlobalFrame.x();
			float dy=segDirMB4inGlobalFrame.y();
			//			float dz=segDirMB4inGlobalFrame.z();
			
			float dx3=segDirMB3inGlobalFrame.x();
			float dy3=segDirMB3inGlobalFrame.y();
			//	float dz3=segDirMB3inGlobalFrame.z();
			
			double cosAng=fabs(dx*dx3+dy*dy3/sqrt((dx3*dx3+dy3*dy3)*(dx*dx+dy*dy)));
			
			if(cosAng>MinCosAng){
			  if(dtSector==13){
			    dtSector=4;
			  }
			  if(dtSector==14){
			    dtSector=10;
			  }
			  
			  std::set<RPCDetId> rollsForThisDT = rollstoreDT[DTStationIndex(0,dtWheel,dtSector,dtStation)]; //It should be always 4
			  
			  assert(rollsForThisDT.size()>=1);
			  
			  for (std::set<RPCDetId>::iterator iteraRoll=rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
			    const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll); //roll asociado a MB4
			    const BoundPlane & RPCSurfaceRB4 = rollasociated->surface(); //surface MB4
			    
			    //   RPCGeomServ rpcsrv(rpcId);
			    // 		    std::string nameRoll = rpcsrv.name();
			    
			    GlobalPoint CenterPointRollGlobal=RPCSurfaceRB4.toGlobal(LocalPoint(0,0,0));
			    LocalPoint CenterRollinMB4Frame = DTSurface4.toLocal(CenterPointRollGlobal); //In MB4
			    LocalPoint segmentPositionMB3inMB4Frame = DTSurface4.toLocal(segmentPositionMB3inGlobal); //In MB4
			    LocalVector segmentDirectionMB3inMB4Frame = DTSurface4.toLocal(segDirMB3inGlobalFrame); //In MB4
			    
			    //The exptrapolation is done in MB4 frame. for local x and z is done from MB4,
			    float Dxz=CenterRollinMB4Frame.z();
			    float Xo4=segmentPositionMB4.x();
			    float dxl=segmentDirectionMB4.x(); //dx local for MB4 segment in MB4 Frame
			    float dzl=segmentDirectionMB4.z(); //dx local for MB4 segment in MB4 Frame
			    
			    float X=Xo4+dxl*Dxz/dzl; //In MB4 frame
			    float Z=Dxz;//In MB4 frame
			    
			    //for local y is done from MB3
			    float Yo34=segmentPositionMB3inMB4Frame.y();
			    float dy34 = segmentDirectionMB3inMB4Frame.y();
			    float dz34 = segmentDirectionMB3inMB4Frame.z();
			    float Dy=Dxz-(segmentPositionMB3inMB4Frame.z()); //Distance beetween the segment in MB3 and the RB4 surface
			    
			    float Y=Yo34+dy34*Dy/dz34;//In MB4 Frame
			    
			    const RectangularStripTopology* top_
			      =dynamic_cast<const RectangularStripTopology*>(&(rollasociated->topology())); //Topology roll asociated MB4
			    LocalPoint xmin = top_->localPosition(0.);
			    LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
			    float rsize = fabs( xmax.x()-xmin.x() );
			    float stripl = top_->stripLength();
			    float stripw = top_->pitch();
			    
			    float extrapolatedDistance = sqrt((Y-Yo34)*(Y-Yo34)+Dy*Dy);
			    
			    if(extrapolatedDistance<=MaxDrb4){ 
			      
			      GlobalPoint GlobalPointExtrapolated = DTSurface4.toGlobal(LocalPoint(X,Y,Z));
			      LocalPoint PointExtrapolatedRPCFrame = RPCSurfaceRB4.toLocal(GlobalPointExtrapolated);
			      
			      if(fabs(PointExtrapolatedRPCFrame.z()) < 5.  &&
				 fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 &&
				 fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){
				
				RPCDetId  rollId = rollasociated->id();
				const float stripPredicted=rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
				
				LogDebug("rpcefficiency")<<"MB4 \t \t \t \t Candidate (from DT Segment) STRIP---> "<<stripPredicted<< std::endl;
				//--------- HISTOGRAM STRIP PREDICTED FROM DT  MB4 -------------------
				
				std::map<std::string, MonitorElement*> meMap=meCollection[rollId.rawId()];				
				meIdDT.str("");
				meIdDT<<"ExpectedOccupancyFromDT_"<<rollId.rawId();
				meMap[meIdDT.str()]->Fill(stripPredicted);
				
				//-------RecHitPart Just For Residual--------
				int countRecHits = 0;
				int cluSize = 0;
				float minres = 3000.;
				
				LogDebug("rpcefficiency")<<"MB4 \t \t \t \t Getting RecHits in Roll Asociated";
				typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
				rangeRecHits recHitCollection =  rpcHits->get(rollasociated->id());
				RPCRecHitCollection::const_iterator recHit;
				
				for (recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++) {
				  countRecHits++;
				  LocalPoint recHitPos=recHit->localPosition();
				  float res=PointExtrapolatedRPCFrame.x()- recHitPos.x();	    
				  LogDebug("rpcefficiency")<<"DT  \t \t \t \t \t Found Rec Hit at "<<res<<"cm of the prediction.";
				  if(fabs(res)<fabs(minres)){
				    minres=res;
				    cluSize = recHit->clusterSize();
				  }
				}		
				
				if(countRecHits==0){
				  LogDebug("rpcefficiency") <<"MB4 \t \t \t \t \t \t THIS ROLL DOESN'T HAVE ANY RECHIT";
				}else{     
				  assert(minres!=3000); 
				  
				  if(fabs(minres)<=(rangestrips+cluSize*0.5)*stripw){
				    assert(rollId.station()==4);
				    if(cluSize==1*dupli){ hGlobalResClu1La[5]->Fill(minres);}
				    else if(cluSize==2*dupli){ hGlobalResClu2La[5]->Fill(minres);}
				    else if(cluSize==3*dupli){ hGlobalResClu3La[5]->Fill(minres);}
				    
				    meIdRPC.str("");
				    meIdRPC<<"RPCDataOccupancyFromDT_"<<rollId.rawId();
				    meMap[meIdRPC.str()]->Fill(stripPredicted);
				  }
				}
			      }else{
				 LogDebug("rpcefficiency")<<"MB4 \t \t \t \t No the prediction is outside of this roll";
			      }
			    }//Condition for the right match
			    else{
			       LogDebug("rpcefficiency")<<"MB4 \t \t \t No, Exrtrapolation too long!, canceled";
			    }
			  }//loop over all the rollsasociated
			}else{
			   LogDebug("rpcefficiency")<<"MB4 \t \t \t \t I found segments in MB4 and MB3 adjacent or same wheel and sector but not compatibles Diferent Directions";
			}
		      }else{//if dtid3.station()==3&&dtid3.sector()==DTId.sector()&&dtid3.wheel()==DTId.wheel()&&segMB3->dim()==4
			 LogDebug("rpcefficiency")<<"MB4 \t \t \t No the same station or same wheel or segment dim in mb3 not 4D";
		      }
		    }//loop over all the segments looking for one in MB3 
		  }else{
		     LogDebug("rpcefficiency")<<"MB4 \t \t \t Is NOT a 2D Segment";
		  }
	  }else{
	     LogDebug("rpcefficiency")<<"MB4 \t \t \t \t There is not just one segment or is not in station 4";
	  }//De aca para abajo esta en dtpart.inl
	}
      }else{
	 LogDebug("rpcefficiency")<<"MB4 \t This event doesn't have 4D Segment";
      }
  }
  
  
  if(inclcsc){
    LogDebug("rpcefficiency") <<"\t Getting the CSC Segments";
    edm::Handle<CSCSegmentCollection> allCSCSegments;
    
    iEvent.getByToken(cscSegments, allCSCSegments);
    
    if(allCSCSegments.isValid()){ 
      if(allCSCSegments->size()>0){
	statistics->Fill(18);
	
	LogDebug("rpcefficiency")<<"CSC \t Number of CSC Segments in this event = "<<allCSCSegments->size();
	
	std::map<CSCDetId,int> CSCSegmentsCounter;
	CSCSegmentCollection::const_iterator segment;
	
	int segmentsInThisEventInTheEndcap=0;
	
	for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
	  CSCSegmentsCounter[segment->cscDetId()]++;
	  segmentsInThisEventInTheEndcap++;
	}    
	
	statistics->Fill(allCSCSegments->size()+18);
	
	 LogDebug("rpcefficiency")<<"CSC \t loop over all the CSCSegments ";
	for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
	  CSCDetId CSCId = segment->cscDetId();
	  
	  if(CSCSegmentsCounter[CSCId]==1 && CSCId.station()!=4 && CSCId.ring()!=1 && allCSCSegments->size()>=2){
	     LogDebug("rpcefficiency")<<"CSC \t \t yes";
	    int cscEndCap = CSCId.endcap();
	    int cscStation = CSCId.station();
	    int cscRing = CSCId.ring();
	    //	    int cscChamber = CSCId.chamber();
	    int rpcRegion = 1; if(cscEndCap==2) rpcRegion= -1;//Relation among the endcaps
	    int rpcRing = cscRing;
	    if(cscRing==4)rpcRing =1;
	    int rpcStation = cscStation;
	    int rpcSegment = CSCId.chamber();
	    
	    LocalPoint segmentPosition= segment->localPosition();
	    LocalVector segmentDirection=segment->localDirection();
	    float dz=segmentDirection.z();
	    
	     LogDebug("rpcefficiency")<<"CSC \t \t Is a good Segment? dim = 4, 4 <= nRecHits <= 10 Incident angle int range 45 < "<<acos(dz)*180/3.1415926<<" < 135? ";
	    
	    if(segment->dimension()==4 && (segment->nRecHits()<=10 && segment->nRecHits()>=4)&& acos(dz)*180/3.1415926 > 45. && acos(dz)*180/3.1415926 < 160. ){ 
	      
	      float Xo=segmentPosition.x();
	      float Yo=segmentPosition.y();
	      float Zo=segmentPosition.z();
	      float dx=segmentDirection.x();
	      float dy=segmentDirection.y();
	      float dz=segmentDirection.z();
	      
	      
	       LogDebug("rpcefficiency")<<"CSC \t \t Getting chamber from Geometry";
	      const CSCChamber* TheChamber=cscGeo->chamber(CSCId); 
	       LogDebug("rpcefficiency")<<"CSC \t \t Getting ID from Chamber";
	      const CSCDetId TheId=TheChamber->id();
	       LogDebug("rpcefficiency")<<"CSC \t \t Printing The Id"<<TheId;
	      std::set<RPCDetId> rollsForThisCSC = rollstoreCSC[CSCStationIndex(rpcRegion,rpcStation,rpcRing,rpcSegment)];
	       LogDebug("rpcefficiency")<<"CSC \t \t Number of rolls for this CSC = "<<rollsForThisCSC.size();
	      
	      if(rpcRing!=1){
		
		//Loop over all the rolls
		for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisCSC.begin();iteraRoll != rollsForThisCSC.end(); iteraRoll++){
		  
		  const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
		  RPCDetId rpcId = rollasociated->id();
		  
		  const BoundPlane & RPCSurface = rollasociated->surface(); 
		  
		  GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
		  LocalPoint CenterRollinCSCFrame = TheChamber->toLocal(CenterPointRollGlobal);
		  
		  float D=CenterRollinCSCFrame.z();
	  	  
		  float X=Xo+dx*D/dz;
		  float Y=Yo+dy*D/dz;
		  float Z=D;
		  
		  const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(rollasociated->topology()));
		  LocalPoint xmin = top_->localPosition(0.);
		   LogDebug("rpcefficiency")<<"CSC \t \t \t xmin of this  Roll "<<xmin<<"cm";
		  LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
		   LogDebug("rpcefficiency")<<"CSC \t \t \t xmax of this  Roll "<<xmax<<"cm";
		  float rsize = fabs( xmax.x()-xmin.x() );
		   LogDebug("rpcefficiency")<<"CSC \t \t \t Roll Size "<<rsize<<"cm";
		  float stripl = top_->stripLength();
		  float stripw = top_->pitch();
		  
		  
		  float extrapolatedDistance = sqrt((X-Xo)*(X-Xo)+(Y-Yo)*(Y-Yo)+(Z-Zo)*(Z-Zo));
		  
		  
		  if(extrapolatedDistance<=MaxD){ 
		    
		    GlobalPoint GlobalPointExtrapolated=TheChamber->toGlobal(LocalPoint(X,Y,Z));
		    LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);
		    
		    
		    if(fabs(PointExtrapolatedRPCFrame.z()) < 10. && 
		       fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 && 
		       fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
		      
		      RPCDetId  rollId = rollasociated->id();
		      RPCGeomServ rpcsrv(rollId);
		      std::string nameRoll = rpcsrv.name();
		      
		       LogDebug("rpcefficiency")<<"CSC \t \t \t \t The RPCName is "<<nameRoll;
		      
		      const float stripPredicted = 
			rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		      
		       LogDebug("rpcefficiency")<<"CSC  \t \t \t \t \t Candidate"<<rollId<<" "<<"(from CSC Segment) STRIP---> "<<stripPredicted<< std::endl;
		      //--------- HISTOGRAM STRIP PREDICTED FROM CSC  -------------------
		      
		      std::map<std::string, MonitorElement*> meMap=meCollection[rpcId.rawId()];
		      meIdCSC.str("");
		      meIdCSC<<"ExpectedOccupancyFromCSC_"<<rollId.rawId();
		      meMap[meIdCSC.str()]->Fill(stripPredicted);
		      //--------------------------------------------------------------------
		      
		      
		      //-------RecHitPart Just For Residual--------
		      int cluSize = 0;
		      int countRecHits = 0;
		      float minres = 3000.;
		      
		       LogDebug("rpcefficiency")<<"CSC  \t \t \t \t \t Getting RecHits in Roll Asociated";
		      typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
		      rangeRecHits recHitCollection =  rpcHits->get(rollasociated->id());
		      RPCRecHitCollection::const_iterator recHit;
		      
		      for (recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++) {
			
			countRecHits++;
			LocalPoint recHitPos=recHit->localPosition();
			float res=PointExtrapolatedRPCFrame.x()- recHitPos.x();
			 LogDebug("rpcefficiency")<<"CSC  \t \t \t \t \t \t Found Rec Hit at "<<res<<"cm of the prediction.";
			if(fabs(res)<fabs(minres)){
			  minres=res;
			  cluSize = recHit->clusterSize();
			   LogDebug("rpcefficiency")<<"CSC  \t \t \t \t \t \t \t New Min Res "<<res<<"cm.";
			}
		      }
		      
		      if(countRecHits==0){
			 LogDebug("rpcefficiency") <<"CSC \t \t \t \t \t THIS ROLL DOESN'T HAVE ANY RECHIT";
		      }else{  
			assert(minres!=3000); 
			
			if(fabs(minres)<=(rangestrips+cluSize*0.5)*stripw){
			   LogDebug("rpcefficiency")<<"CSC  \t \t \t \t \t \t True!";
			  
			  if(rollId.ring()==2&&rollId.roll()==1){
			    if(cluSize==1*dupli) hGlobalResClu1R2A->Fill(minres); 
			    else if(cluSize==2*dupli) hGlobalResClu2R2A->Fill(minres); 
			    else if(cluSize==3*dupli) hGlobalResClu3R2A->Fill(minres);
			  }
			  else if(rollId.ring()==2&&rollId.roll()==2){
			    if(cluSize==1*dupli) hGlobalResClu1R2B->Fill(minres); 
			    else if(cluSize==2*dupli) hGlobalResClu2R2B->Fill(minres); 
			    else if(cluSize==3*dupli) hGlobalResClu3R2B->Fill(minres);
			  }
			  else if(rollId.ring()==2&&rollId.roll()==3){
			    if(cluSize==1*dupli) hGlobalResClu1R2C->Fill(minres); 
			    else if(cluSize==2*dupli) hGlobalResClu2R2C->Fill(minres); 
			    else if(cluSize==3*dupli) hGlobalResClu3R2C->Fill(minres);
			  }
			  else if(rollId.ring()==3&&rollId.roll()==1){
			    if(cluSize==1*dupli) hGlobalResClu1R3A->Fill(minres); 
			    else if(cluSize==2*dupli) hGlobalResClu2R3A->Fill(minres); 
			    else if(cluSize==3*dupli) hGlobalResClu3R3A->Fill(minres);
			  }
			  else if(rollId.ring()==3&&rollId.roll()==2){
			    if(cluSize==1*dupli) hGlobalResClu1R3B->Fill(minres); 
			    else if(cluSize==2*dupli) hGlobalResClu2R3B->Fill(minres); 
			    else if(cluSize==3*dupli) hGlobalResClu3R3B->Fill(minres);
			  }
			  else if(rollId.ring()==3&&rollId.roll()==3){if(cluSize==1*dupli) hGlobalResClu1R3C->Fill(minres); if(cluSize==2*dupli) hGlobalResClu2R3C->Fill(minres); if(cluSize==3*dupli) hGlobalResClu3R3C->Fill(minres);
			  }
			  meIdRPC.str("");
			  meIdRPC<<"RPCDataOccupancyFromCSC_"<<rollId.rawId();
			  meMap[meIdRPC.str()]->Fill(stripPredicted);
			}
		      }

		    }else{
		       LogDebug("rpcefficiency")<<"CSC \t \t \t \t No the prediction is outside of this roll";
		    }//Condition for the right match
		  }else{//if extrapolation distance D is not too long
		     LogDebug("rpcefficiency")<<"CSC \t \t \t No, Exrtrapolation too long!, canceled";
		  }//D so big
		}//loop over the rolls asociated 
	      }//Condition over the LS1 geometry!!!!
	    }//Is the segment 4D?
	  }else{
	     LogDebug("rpcefficiency")<<"CSC \t \t More than one segment in this chamber, or we are in Station Ring 1 or in Station 4";
	  }
	}
      }else{
	 LogDebug("rpcefficiency")<<"CSC This Event doesn't have any CSCSegment";
      }
    }
    }
    
}



