/***************************************
Author: 
Camilo Carrillo
Universidad de los Andes Bogota Colombia
camilo.carrilloATcern.ch
****************************************/

#include "DQMOffline/Muon/interface/RPCEfficiency.h"
#include <memory>
#include "FWCore/Framework/interface/MakerMacros.h"
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
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

int mySegment(RPCDetId rpcId){
  int seg=0;
  int nsec=36;
  int nsub=6;
  if (rpcId.ring()==1 && rpcId.station() > 1) {
    nsub=3;
    nsec=18;
  }
  seg =rpcId.subsector()+nsub*(rpcId.sector()-1);
  if(seg==nsec+1)seg=1;
  return seg;
}

void RPCEfficiency::beginJob(){
  
}

RPCEfficiency::RPCEfficiency(const edm::ParameterSet& iConfig){
  incldt=iConfig.getUntrackedParameter<bool>("incldt",true);
  incldtMB4=iConfig.getUntrackedParameter<bool>("incldtMB4",true);
  inclcsc=iConfig.getUntrackedParameter<bool>("inclcsc",true);
  debug=iConfig.getUntrackedParameter<bool>("debug",false);
  inves=iConfig.getUntrackedParameter<bool>("inves");
 
  rangestrips = iConfig.getUntrackedParameter<double>("rangestrips",1.);
  rangestripsRB4=iConfig.getUntrackedParameter<double>("rangestripsRB4",4.);
  dupli = iConfig.getUntrackedParameter<int>("DuplicationCorrection",2); 
  MinCosAng=iConfig.getUntrackedParameter<double>("MinCosAng",0.9999);
  MaxD=iConfig.getUntrackedParameter<double>("MaxD",20.);
  MaxDrb4=iConfig.getUntrackedParameter<double>("MaxDrb4",30.);
  muonRPCDigis=iConfig.getUntrackedParameter<std::string>("muonRPCDigis","muonRPCDigis");
  cscSegments=iConfig.getUntrackedParameter<std::string>("cscSegments","cscSegments");
  dt4DSegments=iConfig.getUntrackedParameter<std::string>("dt4DSegments","dt4DSegments");

  nameInLog = iConfig.getUntrackedParameter<std::string>("moduleLogName", "RPC_Eff");
  EffSaveRootFile  = iConfig.getUntrackedParameter<bool>("EffSaveRootFile", false); 
  EffRootFileName  = iConfig.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiency.root"); 

  //Interface

  dbe = edm::Service<DQMStore>().operator->();
  
  std::string folder = "Muons/MuonSegEff/";
  dbe->setCurrentFolder(folder);
  statistics = dbe->book1D("Statistics","All Statistics",33,0.5,33.5);
  
  if(debug) std::cout<<"booking Global histograms"<<std::endl;

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
  
  if(debug) std::cout<<"booking Global histograms Change statistics"<<std::endl;

  folder = "Muons/MuonSegEff/Residuals/Investigation";
  dbe->setCurrentFolder(folder);

  //Paper TH1Fs

  DistBorderClu1La1 = dbe->book1D("DistBorderClu1La1","Distance to the Border of the Strip Layer 1 Cluster Size 1",50,-2.,3.);
  DistBorderClu1La2 = dbe->book1D("DistBorderClu1La2","Distance to the Border of the Strip Layer 2 Cluster Size 1",50,-2.,3.);
  DistBorderClu1La3 = dbe->book1D("DistBorderClu1La3","Distance to the Border of the Strip Layer 3 Cluster Size 1",50,-2.,3.);
  DistBorderClu1La4 = dbe->book1D("DistBorderClu1La4","Distance to the Border of the Strip Layer 4 Cluster Size 1",50,-2.,3.);
  DistBorderClu1La5 = dbe->book1D("DistBorderClu1La5","Distance to the Border of the Strip Layer 5 Cluster Size 1",50,-2.,3.);
  DistBorderClu1La6 = dbe->book1D("DistBorderClu1La6","Distance to the Border of the Strip Layer 6 Cluster Size 1",50,-2.,3.);

  DistBorderClu2La1 = dbe->book1D("DistBorderClu2La1","Distance to the Border of the Strip Layer 1 Cluster Size 2",50,-2.,3.);
  DistBorderClu2La2 = dbe->book1D("DistBorderClu2La2","Distance to the Border of the Strip Layer 2 Cluster Size 2",50,-2.,3.);
  DistBorderClu2La3 = dbe->book1D("DistBorderClu2La3","Distance to the Border of the Strip Layer 3 Cluster Size 2",50,-2.,3.);
  DistBorderClu2La4 = dbe->book1D("DistBorderClu2La4","Distance to the Border of the Strip Layer 4 Cluster Size 2",50,-2.,3.);
  DistBorderClu2La5 = dbe->book1D("DistBorderClu2La5","Distance to the Border of the Strip Layer 5 Cluster Size 2",50,-2.,3.);
  DistBorderClu2La6 = dbe->book1D("DistBorderClu2La6","Distance to the Border of the Strip Layer 6 Cluster Size 2",50,-2.,3.);

  DistBorderClu3La1 = dbe->book1D("DistBorderClu3La1","Distance to the Border of the Strip Layer 1 Cluster Size 3",50,-2.,3.);
  DistBorderClu3La2 = dbe->book1D("DistBorderClu3La2","Distance to the Border of the Strip Layer 2 Cluster Size 3",50,-2.,3.);
  DistBorderClu3La3 = dbe->book1D("DistBorderClu3La3","Distance to the Border of the Strip Layer 3 Cluster Size 3",50,-2.,3.);
  DistBorderClu3La4 = dbe->book1D("DistBorderClu3La4","Distance to the Border of the Strip Layer 4 Cluster Size 3",50,-2.,3.);
  DistBorderClu3La5 = dbe->book1D("DistBorderClu3La5","Distance to the Border of the Strip Layer 5 Cluster Size 3",50,-2.,3.);
  DistBorderClu3La6 = dbe->book1D("DistBorderClu3La6","Distance to the Border of the Strip Layer 6 Cluster Size 3",50,-2.,3.);
  
  //Ang Dependence
  float pi = 3.14159265;
  AngClu1La1 = dbe->book1D("AngClu1La1","Angle of incident Muon Layer 1 Cluster Size 1",50,0.,pi);
  AngClu1La2 = dbe->book1D("AngClu1La2","Angle of incident Muon Layer 2 Cluster Size 1",50,0.,pi);
  AngClu1La3 = dbe->book1D("AngClu1La3","Angle of incident Muon Layer 3 Cluster Size 1",50,0.,pi);
  AngClu1La4 = dbe->book1D("AngClu1La4","Angle of incident Muon Layer 4 Cluster Size 1",50,0.,pi);
  AngClu1La5 = dbe->book1D("AngClu1La5","Angle of incident Muon Layer 5 Cluster Size 1",50,0.,pi);
  AngClu1La6 = dbe->book1D("AngClu1La6","Angle of incident Muon Layer 6 Cluster Size 1",50,0.,pi);
  
  AngClu2La1 = dbe->book1D("AngClu2La1","Angle of incident Muon Layer 1 Cluster Size 2",50,0.,pi);
  AngClu2La2 = dbe->book1D("AngClu2La2","Angle of incident Muon Layer 2 Cluster Size 2",50,0.,pi);
  AngClu2La3 = dbe->book1D("AngClu2La3","Angle of incident Muon Layer 3 Cluster Size 2",50,0.,pi);
  AngClu2La4 = dbe->book1D("AngClu2La4","Angle of incident Muon Layer 4 Cluster Size 2",50,0.,pi);
  AngClu2La5 = dbe->book1D("AngClu2La5","Angle of incident Muon Layer 5 Cluster Size 2",50,0.,pi);
  AngClu2La6 = dbe->book1D("AngClu2La6","Angle of incident Muon Layer 6 Cluster Size 2",50,0.,pi);
  
  AngClu3La1 = dbe->book1D("AngClu3La1","Angle of incident Muon Layer 1 Cluster Size 3",50,0.,pi);
  AngClu3La2 = dbe->book1D("AngClu3La2","Angle of incident Muon Layer 2 Cluster Size 3",50,0.,pi);
  AngClu3La3 = dbe->book1D("AngClu3La3","Angle of incident Muon Layer 3 Cluster Size 3",50,0.,pi);
  AngClu3La4 = dbe->book1D("AngClu3La4","Angle of incident Muon Layer 4 Cluster Size 3",50,0.,pi);
  AngClu3La5 = dbe->book1D("AngClu3La5","Angle of incident Muon Layer 5 Cluster Size 3",50,0.,pi);
  AngClu3La6 = dbe->book1D("AngClu3La6","Angle of incident Muon Layer 6 Cluster Size 3",50,0.,pi);

  folder = "Muons/MuonSegEff/Residuals/Barrel";
  dbe->setCurrentFolder(folder);

  //Barrel
  hGlobalResClu1La1 = dbe->book1D("GlobalResidualsClu1La1","RPC Residuals Layer 1 Cluster Size 1",101,-10.,10.);
  hGlobalResClu1La2 = dbe->book1D("GlobalResidualsClu1La2","RPC Residuals Layer 2 Cluster Size 1",101,-10.,10.);
  hGlobalResClu1La3 = dbe->book1D("GlobalResidualsClu1La3","RPC Residuals Layer 3 Cluster Size 1",101,-10.,10.);
  hGlobalResClu1La4 = dbe->book1D("GlobalResidualsClu1La4","RPC Residuals Layer 4 Cluster Size 1",101,-10.,10.);
  hGlobalResClu1La5 = dbe->book1D("GlobalResidualsClu1La5","RPC Residuals Layer 5 Cluster Size 1",101,-10.,10.);
  hGlobalResClu1La6 = dbe->book1D("GlobalResidualsClu1La6","RPC Residuals Layer 6 Cluster Size 1",101,-10.,10.);

  hGlobalResClu2La1 = dbe->book1D("GlobalResidualsClu2La1","RPC Residuals Layer 1 Cluster Size 2",101,-10.,10.);
  hGlobalResClu2La2 = dbe->book1D("GlobalResidualsClu2La2","RPC Residuals Layer 2 Cluster Size 2",101,-10.,10.);
  hGlobalResClu2La3 = dbe->book1D("GlobalResidualsClu2La3","RPC Residuals Layer 3 Cluster Size 2",101,-10.,10.);
  hGlobalResClu2La4 = dbe->book1D("GlobalResidualsClu2La4","RPC Residuals Layer 4 Cluster Size 2",101,-10.,10.);
  hGlobalResClu2La5 = dbe->book1D("GlobalResidualsClu2La5","RPC Residuals Layer 5 Cluster Size 2",101,-10.,10.);
  hGlobalResClu2La6 = dbe->book1D("GlobalResidualsClu2La6","RPC Residuals Layer 6 Cluster Size 2",101,-10.,10.);

  hGlobalResClu3La1 = dbe->book1D("GlobalResidualsClu3La1","RPC Residuals Layer 1 Cluster Size 3",101,-10.,10.);
  hGlobalResClu3La2 = dbe->book1D("GlobalResidualsClu3La2","RPC Residuals Layer 2 Cluster Size 3",101,-10.,10.);
  hGlobalResClu3La3 = dbe->book1D("GlobalResidualsClu3La3","RPC Residuals Layer 3 Cluster Size 3",101,-10.,10.);
  hGlobalResClu3La4 = dbe->book1D("GlobalResidualsClu3La4","RPC Residuals Layer 4 Cluster Size 3",101,-10.,10.);
  hGlobalResClu3La5 = dbe->book1D("GlobalResidualsClu3La5","RPC Residuals Layer 5 Cluster Size 3",101,-10.,10.);
  hGlobalResClu3La6 = dbe->book1D("GlobalResidualsClu3La6","RPC Residuals Layer 6 Cluster Size 3",101,-10.,10.);

  if(debug) std::cout<<"Booking Residuals for EndCap"<<std::endl;
  folder = "Muons/MuonSegEff/Residuals/EndCap";
  dbe->setCurrentFolder(folder);

  //Endcap  
  hGlobalResClu1R3C = dbe->book1D("GlobalResidualsClu1R3C","RPC Residuals Ring 3 Roll C Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R3B = dbe->book1D("GlobalResidualsClu1R3B","RPC Residuals Ring 3 Roll B Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R3A = dbe->book1D("GlobalResidualsClu1R3A","RPC Residuals Ring 3 Roll A Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R2C = dbe->book1D("GlobalResidualsClu1R2C","RPC Residuals Ring 2 Roll C Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R2B = dbe->book1D("GlobalResidualsClu1R2B","RPC Residuals Ring 2 Roll B Cluster Size 1",101,-10.,10.);
  hGlobalResClu1R2A = dbe->book1D("GlobalResidualsClu1R2A","RPC Residuals Ring 2 Roll A Cluster Size 1",101,-10.,10.);

  hGlobalResClu2R3C = dbe->book1D("GlobalResidualsClu2R3C","RPC Residuals Ring 3 Roll C Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R3B = dbe->book1D("GlobalResidualsClu2R3B","RPC Residuals Ring 3 Roll B Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R3A = dbe->book1D("GlobalResidualsClu2R3A","RPC Residuals Ring 3 Roll A Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R2C = dbe->book1D("GlobalResidualsClu2R2C","RPC Residuals Ring 2 Roll C Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R2B = dbe->book1D("GlobalResidualsClu2R2B","RPC Residuals Ring 2 Roll B Cluster Size 2",101,-10.,10.);
  hGlobalResClu2R2A = dbe->book1D("GlobalResidualsClu2R2A","RPC Residuals Ring 2 Roll A Cluster Size 2",101,-10.,10.);

  hGlobalResClu3R3C = dbe->book1D("GlobalResidualsClu3R3C","RPC Residuals Ring 3 Roll C Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R3B = dbe->book1D("GlobalResidualsClu3R3B","RPC Residuals Ring 3 Roll B Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R3A = dbe->book1D("GlobalResidualsClu3R3A","RPC Residuals Ring 3 Roll A Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R2C = dbe->book1D("GlobalResidualsClu3R2C","RPC Residuals Ring 2 Roll C Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R2B = dbe->book1D("GlobalResidualsClu3R2B","RPC Residuals Ring 2 Roll B Cluster Size 3",101,-10.,10.);
  hGlobalResClu3R2A = dbe->book1D("GlobalResidualsClu3R2A","RPC Residuals Ring 2 Roll A Cluster Size 3",101,-10.,10.);

  
  if(debug) ofrej.open("rejected.txt");

  if(debug) std::cout<<"Rejected done"<<std::endl;

}

void RPCEfficiency::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){

  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  iSetup.get<MuonGeometryRecord>().get(dtGeo);
  iSetup.get<MuonGeometryRecord>().get(cscGeo);

  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	int region=rpcId.region();
	//booking all histograms
	RPCGeomServ rpcsrv(rpcId);
	std::string nameRoll = rpcsrv.name();
	//std::cout<<"Booking for "<<nameRoll<<std::endl;
	meCollection[nameRoll] = bookDetUnitSeg(rpcId,(*r)->nstrips());
	
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
	if(region!=0 && inclcsc){
	  //std::cout<<"--Filling the cscstore"<<rpcId<<std::endl;
	  int region=rpcId.region();
          int station=rpcId.station();
          int ring=rpcId.ring();
          int cscring=ring;
          int cscstation=station;
	  RPCGeomServ rpcsrv(rpcId);
	  int rpcsegment = mySegment(rpcId); //This replace rpcsrv.segment();
	  //std::cout<<"My segment="<<mySegment(rpcId)<<" GeomServ="<<rpcsrv.segment()<<std::endl;
	  int cscchamber = rpcsegment;//FIX THIS ACCORDING TO RPCGeomServ::segment()Definition
          if((station==2||station==3)&&ring==3){//Adding Ring 3 of RPC to the CSC Ring 2
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

  //Now filling in order to extrapolate to other wheels.
  /*
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){

      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	
	int region=rpcId.region();
	
	if(region==0&&(incldt||incldtMB4)&&rpcId.ring()!=0&&rpcId.station()!=4){
	  //std::cout<<"--Filling the dtstore for statistics"<<rpcId<<std::endl;
	  
	  int sidewheel = 0;
	  
	  if(rpcId.ring()==-2){
	    sidewheel=-1;
	  }
	  else if(rpcId.ring()==-1){
	    sidewheel=0;
	  }
	  else if(rpcId.ring()==1){
	    sidewheel=0;
	  }
	  else if(rpcId.ring()==2){
	    sidewheel=1;
	  }
	  int wheel= sidewheel;
	  int sector=rpcId.sector();
	  int station=rpcId.station();
	  DTStationIndex ind(region,wheel,sector,station);
	  std::set<RPCDetId> myrolls;
	  if (rollstoreDT.find(ind)!=rollstoreDT.end()) myrolls=rollstoreDT[ind];
	  myrolls.insert(rpcId);
	  rollstoreDT[ind]=myrolls;
	}
	if(region!=0 && inclcsc && (rpcId.ring()==2 || rpcId.ring()==3)){
	  int region=rpcId.region();
          int station=rpcId.station();
          int ring=rpcId.ring();
	  int sidering = 0;
	  if(ring==2)sidering =3;
	  else if(ring==3) sidering =2;
	  
          int cscring=sidering;
          int cscstation=station;
	  RPCGeomServ rpcsrv(rpcId);
	  int rpcsegment = mySegment(rpcId);
	  
	  if((station==2||station==3)&&ring==3) cscring = 2; //CSC Ring 2 covers rpc ring 2 & 3
          	  
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
	  if (rollstoreCSC.find(indDos)!=rollstoreCSC.end())myrollsDos=rollstoreCSC[indDos];          
          myrollsDos.insert(rpcId);
          rollstoreCSC[indDos]=myrolls;
	  

        }
      }
    }
  }
  */
  //booking global histograms

 
}//beginRun


RPCEfficiency::~RPCEfficiency()
{

}

void RPCEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  statistics->Fill(1);
  using namespace edm;
  
  char layerLabel[128];
  char meIdRPC [128];
  char meIdDT [128];
  char meIdCSC [128];

  //-------------Filling Other Histograms for correlations -----------
  
  if(debug) std::cout <<"Digi Getting the RPC Digis"<<std::endl;
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel(muonRPCDigis, rpcDigis);
  char detUnitLabel[128];
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	RPCGeomServ rpcsrv(rpcId);
	std::string nameRoll = rpcsrv.name();
	std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
	sprintf(detUnitLabel ,"%s",nameRoll.c_str());

	sprintf(layerLabel ,"%s",nameRoll.c_str());
	RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rpcId);
	for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
	  int stripDetected=digiIt->strip();
	  sprintf(meIdRPC,"BXDistribution_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(digiIt->bx());
	  if(rpcId.region()==0){
	    sprintf(meIdRPC,"RealDetectedOccupancyFromDT_%s",detUnitLabel);
	  }else {
	    sprintf(meIdRPC,"RealDetectedOccupancyFromCSC_%s",detUnitLabel);
	  }
	  meMap[meIdRPC]->Fill(stripDetected); //have a look to this!
	  
	}
      }
    }
  }
  
  //------------------------------------------------------------------------------------
  
  if(debug) std::cout <<"\t Getting the RPC RecHits"<<std::endl;
  Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByType(rpcHits);

  if(incldt){
    if(debug) std::cout<<"\t Getting the DT Segments"<<std::endl;
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    iEvent.getByLabel(dt4DSegments, all4DSegments);
    if(all4DSegments->size()>0){
      if(all4DSegments->size()<=16) statistics->Fill(2);

      if(debug) std::cout<<"\t Number of DT Segments in this event = "<<all4DSegments->size()<<std::endl;
  
      std::map<DTChamberId,int> DTSegmentCounter;
      DTRecSegment4DCollection::const_iterator segment;  
  
      for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
	DTSegmentCounter[segment->chamberId()]++;
      }    
  
      statistics->Fill(all4DSegments->size()+2);

      if(debug) std::cout<<"\t Loop over all the 4D Segments"<<std::endl;
      for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
    
	DTChamberId DTId = segment->chamberId();

	
	if(debug) std::cout<<"DT  \t \t This Segment is in Chamber id: "<<DTId<<std::endl;
	if(debug) std::cout<<"DT  \t \t Number of segments in this DT = "<<DTSegmentCounter[DTId]<<std::endl;
	if(debug) std::cout<<"DT  \t \t Is the only one in this DT? and is not in the 4th Station?"<<std::endl;

    
	if(DTSegmentCounter[DTId]==1 && DTId.station()!=4){	
 
	  int dtWheel = DTId.wheel();
	  int dtStation = DTId.station();
	  int dtSector = DTId.sector();      

	  LocalPoint segmentPosition= segment->localPosition();
	  LocalVector segmentDirection=segment->localDirection();
      
	  const GeomDet* gdet=dtGeo->idToDet(segment->geographicalId());
	  const BoundPlane & DTSurface = gdet->surface();
      
	  //check if the dimension of the segment is 4 

	  if(debug) std::cout<<"DT  \t \t Is the segment 4D?"<<std::endl;
      
	  if(segment->dimension()==4){

	    if(debug) std::cout<<"DT  \t \t yes"<<std::endl;
	    if(debug) std::cout<<"DT  \t \t DT Segment Dimension "<<segment->dimension()<<std::endl; 
	
	    float Xo=segmentPosition.x();
	    float Yo=segmentPosition.y();
	    float Zo=segmentPosition.z();
	    float dx=segmentDirection.x();
	    float dy=segmentDirection.y();
	    float dz=segmentDirection.z();
	    
	    std::set<RPCDetId> rollsForThisDT = rollstoreDT[DTStationIndex(0,dtWheel,dtSector,dtStation)];

	    if(debug) std::cout<<"DT  \t \t Number of rolls for this DT = "<<rollsForThisDT.size()<<std::endl;
       
	    assert(rollsForThisDT.size()>=1);

	    if(debug) std::cout<<"DT  \t \t Loop over all the rolls asociated to this DT"<<std::endl;
	    for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
	      const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
	      RPCDetId rpcId = rollasociated->id();
	      const BoundPlane & RPCSurface = rollasociated->surface(); 
	      
	      RPCGeomServ rpcsrv(rpcId);
	      std::string nameRoll = rpcsrv.name();

	      if(debug) std::cout<<"DT  \t \t \t RollName: "<<nameRoll<<std::endl;
	      if(debug) std::cout<<"DT  \t \t \t Doing the extrapolation to this roll"<<std::endl;
	      if(debug) std::cout<<"DT  \t \t \t DT Segment Direction in DTLocal "<<segmentDirection<<std::endl;
	      if(debug) std::cout<<"DT  \t \t \t DT Segment Point in DTLocal "<<segmentPosition<<std::endl;
	  
	      GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));

	      LocalPoint CenterRollinDTFrame = DTSurface.toLocal(CenterPointRollGlobal);

	      if(debug) std::cout<<"DT  \t \t \t Center (0,0,0) Roll In DTLocal"<<CenterRollinDTFrame<<std::endl;
	      if(debug) std::cout<<"DT  \t \t \t Center (0,0,0) of the Roll in Global"<<CenterPointRollGlobal<<std::endl;

	      float D=CenterRollinDTFrame.z();
	  
	      float X=Xo+dx*D/dz;
	      float Y=Yo+dy*D/dz;
	      float Z=D;
	
	      const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&(rollasociated->topology()));
	      LocalPoint xmin = top_->localPosition(0.);
	      if(debug) std::cout<<"DT  \t \t \t xmin of this  Roll "<<xmin<<"cm"<<std::endl;
	      LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
	      if(debug) std::cout<<"DT  \t \t \t xmax of this  Roll "<<xmax<<"cm"<<std::endl;
	      float rsize = fabs( xmax.x()-xmin.x() );
	      if(debug) std::cout<<"DT  \t \t \t Roll Size "<<rsize<<"cm"<<std::endl;
	      float stripl = top_->stripLength();
	      float stripw = top_->pitch();
	  	  
	      if(debug) std::cout<<"DT  \t \t \t Strip Lenght "<<stripl<<"cm"<<std::endl;
	      if(debug) std::cout<<"DT  \t \t \t Strip Width "<<stripw<<"cm"<<std::endl;
	      if(debug) std::cout<<"DT  \t \t \t X Predicted in DTLocal= "<<X<<"cm"<<std::endl;
	      if(debug) std::cout<<"DT  \t \t \t Y Predicted in DTLocal= "<<Y<<"cm"<<std::endl;
	      if(debug) std::cout<<"DT  \t \t \t Z Predicted in DTLocal= "<<Z<<"cm"<<std::endl;

	      float extrapolatedDistance = sqrt((X-Xo)*(X-Xo)+(Y-Yo)*(Y-Yo)+(Z-Zo)*(Z-Zo));

	      if(debug) std::cout<<"DT  \t \t \t Is the distance of extrapolation less than MaxD? ="<<extrapolatedDistance<<"cm"<<"MaxD="<<MaxD<<"cm"<<std::endl;

	      if(extrapolatedDistance<=MaxD){ 
		if(debug) std::cout<<"DT  \t \t \t yes"<<std::endl;   
		GlobalPoint GlobalPointExtrapolated = DTSurface.toGlobal(LocalPoint(X,Y,Z));
		if(debug) std::cout<<"DT  \t \t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
		LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);
	    
		if(debug) std::cout<<"DT  \t \t \t Point Extrapolated in RPCLocal"<<PointExtrapolatedRPCFrame<< std::endl;
		if(debug) std::cout<<"DT  \t \t \t Corner of the Roll = ("<<rsize*0.5<<","<<stripl*0.5<<")"<<std::endl;
		if(debug) std::cout<<"DT \t \t \t Info About the Point Extrapolated in X Abs ("<<fabs(PointExtrapolatedRPCFrame.x())<<","
				   <<fabs(PointExtrapolatedRPCFrame.y())<<","<<fabs(PointExtrapolatedRPCFrame.z())<<")"<<std::endl;
		if(debug) std::cout<<"DT  \t \t \t Does the extrapolation go inside this roll?"<<std::endl;

		if(fabs(PointExtrapolatedRPCFrame.z()) < 10. && 
		   fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 && 
		   fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){
		  
		  if(debug) std::cout<<"DT  \t \t \t \t yes"<<std::endl;	

		  RPCDetId  rollId = rollasociated->id();
		  
		  RPCGeomServ rpcsrv(rollId);
		  std::string nameRoll = rpcsrv.name();
		  if(debug) std::cout<<"DT  \t \t \t \t The RPCName is "<<nameRoll<<std::endl;		    
		  const float stripPredicted = 
		    rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		  
		  if(debug) std::cout<<"DT  \t \t \t \t Candidate (from DT Segment) STRIP---> "<<stripPredicted<< std::endl;		  
		  //---- HISTOGRAM STRIP PREDICTED FROM DT ----
		  char detUnitLabel[128];
		  sprintf(detUnitLabel ,"%s",nameRoll.c_str());
		  sprintf(layerLabel ,"%s",nameRoll.c_str());
		    
		  std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
		    
		  sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
		  if(debug) std::cout<<"DT \t \t \t \t Filling Expected for "<<meIdDT<<" with "<<stripPredicted<<std::endl;
		  if(fabs(stripPredicted-rollasociated->nstrips())<1.) if(debug) std::cout<<"DT \t \t \t \t Extrapolating near last strip, Event"<<iEvent.id()<<" stripPredicted="<<stripPredicted<<" Number of strips="<<rollasociated->nstrips()<<std::endl;
		  if(fabs(stripPredicted)<1.) if(debug) std::cout<<"DT \t \t \t \t Extrapolating near first strip, Event"<<iEvent.id()<<" stripPredicted="<<stripPredicted<<" Number of strips="<<rollasociated->nstrips()<<std::endl;
		  meMap[meIdDT]->Fill(stripPredicted);
		  //-----------------------------------------------------
		  

		  //-------RecHitPart Just For Residual--------
		  int countRecHits = 0;
		  int cluSize = 0;
		  float minres = 3000.;
		  
		  if(debug) std::cout<<"DT  \t \t \t \t Getting RecHits in Roll Asociated"<<std::endl;
		  typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
		  rangeRecHits recHitCollection =  rpcHits->get(rollasociated->id());
		  RPCRecHitCollection::const_iterator recHit;
		  
		  for (recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++) {
		    countRecHits++;
		    LocalPoint recHitPos=recHit->localPosition();
		    float res=PointExtrapolatedRPCFrame.x()- recHitPos.x();	    
		    if(debug) std::cout<<"DT  \t \t \t \t \t Found Rec Hit at "<<res<<"cm of the prediction."<<std::endl;
		    if(fabs(res)<fabs(minres)){
		      minres=res;
		      cluSize = recHit->clusterSize();
		      if(debug) std::cout<<"DT  \t \t \t \t \t \t New Min Res "<<res<<"cm."<<std::endl;
		    }
		  }
		  
		  bool anycoincidence=false;
	
		  if(countRecHits==0){
		    if(debug) std::cout <<"DT \t \t \t \t \t THIS ROLL DOESN'T HAVE ANY RECHIT"<<std::endl;
		  }else{
		    assert(minres!=3000);     
		    
		    if(debug) std::cout<<"DT  \t \t \t \t \t PointExtrapolatedRPCFrame.x="<<PointExtrapolatedRPCFrame.x()<<" Minimal Residual="<<minres<<std::endl;
		    if(debug) std::cout<<"DT  \t \t \t \t \t Minimal Residual less than stripw*rangestrips? minres="<<minres<<" range="<<rangestrips<<" stripw="<<stripw<<" cluSize"<<cluSize<<" <=compare minres with"<<(rangestrips+cluSize*0.5)*stripw<<std::endl;
		    if(fabs(minres)<=(rangestrips+cluSize*0.5)*stripw){
		      if(debug) std::cout<<"DT  \t \t \t \t \t \t True!"<<std::endl;
		      anycoincidence=true;
		    }
		  }
		  if(anycoincidence){
		    if(debug) std::cout<<"DT  \t \t \t \t \t At least one RecHit inside the range, Predicted="<<stripPredicted<<" minres="<<minres<<"cm range="<<rangestrips<<"strips stripw="<<stripw<<"cm"<<std::endl;
		    if(debug) std::cout<<"DT  \t \t \t \t \t Norm of Cosine Directors="<<dx*dx+dy*dy+dz*dz<<"~1?"<<std::endl;
		    
		    //-----RESIDUALS----------
		    if(inves){
		      float cosal = dx/sqrt(dx*dx+dz*dz);
		      if(debug) std::cout<<"DT \t \t \t \t \t Angle="<<acos(cosal)*180/3.1415926<<" degree"<<std::endl;
		      if(debug) std::cout<<"DT \t \t \t \t \t Filling the Residuals Histogram for globals with "<<minres<<"And the angular incidence with Cos Alpha="<<cosal<<std::endl;
		      if(rollId.station()==1&&rollId.layer()==1)     { if(cluSize==1*dupli) {hGlobalResClu1La1->Fill(minres); AngClu1La1->Fill(acos(cosal)); DistBorderClu1La1->Fill(minres/stripw+0.5);}if(cluSize==2*dupli){ hGlobalResClu2La1->Fill(minres);  AngClu2La1->Fill(acos(cosal));DistBorderClu2La1->Fill(minres/stripw+0.5);} if(cluSize==3*dupli){ hGlobalResClu3La1->Fill(minres);  AngClu3La1->Fill(acos(cosal));} DistBorderClu3La1->Fill(minres/stripw+0.5);}
		      else if(rollId.station()==1&&rollId.layer()==2){ if(cluSize==1*dupli) {hGlobalResClu1La2->Fill(minres); AngClu1La2->Fill(acos(cosal)); DistBorderClu1La2->Fill(minres/stripw+0.5);}if(cluSize==2*dupli){ hGlobalResClu2La2->Fill(minres);  AngClu2La2->Fill(acos(cosal));DistBorderClu2La2->Fill(minres/stripw+0.5);} if(cluSize==3*dupli){ hGlobalResClu3La2->Fill(minres);  AngClu3La2->Fill(acos(cosal));} DistBorderClu3La2->Fill(minres/stripw+0.5);}
		      else if(rollId.station()==2&&rollId.layer()==1){ if(cluSize==1*dupli) {hGlobalResClu1La3->Fill(minres); AngClu1La3->Fill(acos(cosal)); DistBorderClu1La3->Fill(minres/stripw+0.5);}if(cluSize==2*dupli){ hGlobalResClu2La3->Fill(minres);  AngClu2La3->Fill(acos(cosal));DistBorderClu2La3->Fill(minres/stripw+0.5);} if(cluSize==3*dupli){ hGlobalResClu3La3->Fill(minres);  AngClu3La3->Fill(acos(cosal));} DistBorderClu3La3->Fill(minres/stripw+0.5);}
		      else if(rollId.station()==2&&rollId.layer()==2){ if(cluSize==1*dupli) {hGlobalResClu1La4->Fill(minres); AngClu1La4->Fill(acos(cosal)); DistBorderClu1La4->Fill(minres/stripw+0.5);}if(cluSize==2*dupli){ hGlobalResClu2La4->Fill(minres);  AngClu2La4->Fill(acos(cosal));DistBorderClu2La4->Fill(minres/stripw+0.5);} if(cluSize==3*dupli){ hGlobalResClu3La4->Fill(minres);  AngClu3La4->Fill(acos(cosal));} DistBorderClu3La4->Fill(minres/stripw+0.5);}
		      else if(rollId.station()==3)                   { if(cluSize==1*dupli) {hGlobalResClu1La5->Fill(minres); AngClu1La5->Fill(acos(cosal)); DistBorderClu1La5->Fill(minres/stripw+0.5);}if(cluSize==2*dupli){ hGlobalResClu2La5->Fill(minres);  AngClu2La5->Fill(acos(cosal));DistBorderClu2La5->Fill(minres/stripw+0.5);} if(cluSize==3*dupli){ hGlobalResClu3La5->Fill(minres);  AngClu3La5->Fill(acos(cosal));} DistBorderClu3La5->Fill(minres/stripw+0.5);}
		      //------------------------
		    }
		    sprintf(meIdRPC,"RPCDataOccupancyFromDT_%s",detUnitLabel);
		    if(debug) std::cout<<"DT \t \t \t \t \t COINCIDENCE!!! Event="<<iEvent.id()<<" Filling RPC Data Occupancy for "<<meIdRPC<<" with "<<stripPredicted<<std::endl; 
		    meMap[meIdRPC]->Fill(stripPredicted);
		  }
		  else{
		    RPCGeomServ rpcsrv(rollasociated->id());
		    std::string nameRoll = rpcsrv.name();
		    if(debug) std::cout<<"DT \t \t \t \t \t A roll was ineficient in event "<<iEvent.id().event()<<std::endl;
		    if(debug) ofrej<<"DTs \t Wh "<<dtWheel
				   <<"\t St "<<dtStation
				   <<"\t Se "<<dtSector
				   <<"\t Roll "<<nameRoll
				   <<"\t Event "
				   <<iEvent.id().event()
				   <<"\t Run "	
				   <<iEvent.id().run()	
				   <<std::endl;
		  }
		}else {
		  if(debug) std::cout<<"DT \t \t \t \t No the prediction is outside of this roll"<<std::endl;
		}//Condition for the right match
	      }else{
		if(debug) std::cout<<"DT \t \t \t No, Exrtrapolation too long!, canceled"<<std::endl;
	      }//D so big
	    }//loop over all the rolls asociated
	  }//Is the segment 4D?
	}else {
	  if(debug) std::cout<<"DT \t \t More than one segment in this chamber, or we are in Station 4"<<std::endl;
	}
      }
    }
    else {
      if(debug) std::cout<<"DT This Event doesn't have any DT4DDSegment"<<std::endl; //is ther more than 1 segment in this event?
    }
  }
  
  if(incldtMB4){
    if(debug) std::cout <<"MB4 \t Getting ALL the DT Segments"<<std::endl;
    edm::Handle<DTRecSegment4DCollection> all4DSegments;
    iEvent.getByLabel(dt4DSegments, all4DSegments);
    
    if(all4DSegments->size()>0){
  
      std::map<DTChamberId,int> DTSegmentCounter;
      DTRecSegment4DCollection::const_iterator segment;  
  
      for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
	DTSegmentCounter[segment->chamberId()]++;
      }    

      if(debug) std::cout<<"MB4 \t \t Loop Over all4DSegments"<<std::endl;
      for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
    
	DTChamberId DTId = segment->chamberId();

	if(debug) std::cout<<"MB4 \t \t \t Is the only one in the chamber? and is in the Station 4?"<<std::endl;

	if(DTSegmentCounter[DTId] == 1 && DTId.station()==4){

	  if(debug) std::cout<<"MB4 \t \t \t yes"<<std::endl;
	  int dtWheel = DTId.wheel();
	  int dtStation = DTId.station();
	  int dtSector = DTId.sector();
      
	  LocalPoint segmentPosition= segment->localPosition();
	  LocalVector segmentDirection=segment->localDirection();
            
	  //check if the dimension of the segment is 2 and the station is 4
	  
	  
	  if(debug) std::cout<<"MB4 \t \t \t \t The Segment in MB4 is 2D?"<<std::endl;
	  if(segment->dimension()==2){
	    if(debug) std::cout<<"MB4 \t \t \t \t yes"<<std::endl;
	    LocalVector segmentDirectionMB4=segmentDirection;
	    LocalPoint segmentPositionMB4=segmentPosition;
	
	    bool compatiblesegments=false;
	    float dx=segmentDirectionMB4.x();
	    float dz=segmentDirectionMB4.z();
	    
	    const BoundPlane& DTSurface4 = dtGeo->idToDet(DTId)->surface();
	    
	    DTRecSegment4DCollection::const_iterator segMB3;  
	    
	    if(debug) std::cout<<"MB4 \t \t \t \t Loop on segments in =sector && MB3 && adjacent Wheel && y dim=4"<<std::endl;
	    for(segMB3=all4DSegments->begin();segMB3!=all4DSegments->end();++segMB3){
	      
	      DTChamberId dtid3 = segMB3->chamberId();  
	      
	      if(dtid3.sector()==DTId.sector() 
		 && dtid3.station()==3
		 && abs(dtid3.wheel()-DTId.wheel())<2
		 && DTSegmentCounter[dtid3] == 1
		 && segMB3->dimension()==4){

		const GeomDet* gdet3=dtGeo->idToDet(segMB3->geographicalId());
		const BoundPlane & DTSurface3 = gdet3->surface();
	      
		float dx3=segMB3->localDirection().x();
		float dy3=segMB3->localDirection().y();
		float dz3=segMB3->localDirection().z();
	    
		LocalVector segDirMB4inMB3Frame=DTSurface3.toLocal(DTSurface4.toGlobal(segmentDirectionMB4));
		
		double cosAng=fabs(dx*dx3+dz*dz3/sqrt((dx3*dx3+dz3*dz3)*(dx*dx+dz*dz)));
		if(fabs(cosAng)>1.){
		  std::cout<<"dx="<<dx<<" dz="<<dz<<std::endl;
		  std::cout<<"dx3="<<dx3<<" dz3="<<dz<<std::endl;
		  std::cout<<cosAng<<std::endl;
		}
		
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
		    float stripw = top_->pitch();

		    
		    if(debug) std::cout<<"MB4 \t \t \t Strip Lenght "<<stripl<<"cm"<<std::endl;
		    if(debug) std::cout<<"MB4 \t \t \t Strip Width "<<stripw<<"cm"<<std::endl;

		    if(debug) std::cout<<"MB4 \t \t \t X Predicted in MB3Local= "<<X<<"cm"<<std::endl;
		    if(debug) std::cout<<"MB4 \t \t \t Y Predicted in MB3DTLocal= "<<Y<<"cm"<<std::endl;
		    if(debug) std::cout<<"MB4 \t \t \t Z Predicted in MB3DTLocal= "<<Z<<"cm"<<std::endl;

		    float extrapolatedDistance = sqrt((X-Xo3)*(X-Xo3)+(Y-Yo3)*(Y-Yo3)+(Z-Zo3)*(Z-Zo3));

		    if(extrapolatedDistance<=MaxDrb4){ 
		      if(debug) std::cout<<"MB4 \t \t \t yes"<<std::endl;

		      GlobalPoint GlobalPointExtrapolated = DTSurfaceMB3.toGlobal(LocalPoint(X,Y,Z));
		      
		      if(debug) std::cout<<"MB4 \t \t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
		      
		      LocalPoint PointExtrapolatedRPCFrame = RPCSurfaceRB4.toLocal(GlobalPointExtrapolated);

		      if(debug) std::cout<<"MB4 \t \t \t Point Extrapolated in RPCLocal"<<PointExtrapolatedRPCFrame<< std::endl;
		      if(debug) std::cout<<"MB4 \t \t \t Corner of the Roll = ("<<rsize*0.5<<","<<stripl*0.5<<")"<<std::endl;
		      if(debug) std::cout<<"MB4 \t \t \t Info About the Point Extrapolated in X Abs ("<<fabs(PointExtrapolatedRPCFrame.x())<<","
					 <<fabs(PointExtrapolatedRPCFrame.y())<<","<<fabs(PointExtrapolatedRPCFrame.z())<<")"<<std::endl;
	
		      if(debug) std::cout<<"MB4 \t \t \t Does the extrapolation go inside this roll?"<<std::endl;
		
		      if(fabs(PointExtrapolatedRPCFrame.z()) < 10.  &&
			 fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 &&
			 fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){

			if(debug) std::cout<<"MB4 \t \t \t \t yes"<<std::endl;
			
			RPCDetId  rollId = rollasociated->id();

			RPCGeomServ rpcsrv(rollId);
			std::string nameRoll = rpcsrv.name();
			if(debug) std::cout<<"MB4 \t \t \t \t \t The RPCName is "<<nameRoll<<std::endl;
			const float stripPredicted=
			  rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 
		  
			if(debug) std::cout<<"MB4 \t \t \t \t Candidate (from DT Segment) STRIP---> "<<stripPredicted<< std::endl;
			//--------- HISTOGRAM STRIP PREDICTED FROM DT  MB4 -------------------
			char detUnitLabel[128];
			sprintf(detUnitLabel ,"%s",nameRoll.c_str());
			sprintf(layerLabel ,"%s",nameRoll.c_str());
			
			std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
			
			if(debug) std::cout<<"MB4 \t \t \t \t \t Filling Expected"<<std::endl;
			sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
			meMap[meIdDT]->Fill(stripPredicted);
			//-------------------------------------------------
			

			//-------RecHitPart Just For Residual--------
			int countRecHits = 0;
			int cluSize = 0;
			float minres = 3000.;
			
			if(debug) std::cout<<"MB4 \t \t \t \t Getting RecHits in Roll Asociated"<<std::endl;
			typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
			rangeRecHits recHitCollection =  rpcHits->get(rollasociated->id());
			RPCRecHitCollection::const_iterator recHit;
			
			for (recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++) {
			  countRecHits++;
			  LocalPoint recHitPos=recHit->localPosition();
			  float res=PointExtrapolatedRPCFrame.x()- recHitPos.x();	    
			  if(debug) std::cout<<"DT  \t \t \t \t \t Found Rec Hit at "<<res<<"cm of the prediction."<<std::endl;
			  if(fabs(res)<fabs(minres)){
			    minres=res;
			    cluSize = recHit->clusterSize();
			  }
			}		

			bool anycoincidence=false;
			
			if(countRecHits==0){
			  if(debug) std::cout <<"MB4 \t \t \t \t \t \t THIS ROLL DOESN'T HAVE ANY RECHIT"<<std::endl;
			}else{     
			  assert(minres!=3000); 

			  if(debug) std::cout<<"MB4 \t \t \t \t \t \t PointExtrapolatedRPCFrame.x="<<PointExtrapolatedRPCFrame.x()<<" Minimal Residual ="<<minres<<std::endl;
			  if(debug) std::cout<<"MB4 \t \t \t \t \t \t Minimal Residual less than stripw*rangestrips? minres="<<minres<<" range="<<rangestrips<<" stripw="<<stripw<<" cluSize"<<cluSize<<" <=compare minres with"<<(rangestrips+cluSize*0.5)*stripw<<std::endl;
			  if(fabs(minres)<=(rangestrips+cluSize*0.5)*stripw){
			    if(debug) std::cout<<"MB4 \t \t \t \t \t \t \t True!"<<std::endl;
			    anycoincidence=true;
			  }
			}
			if(anycoincidence){
			  if(debug) std::cout<<"MB4  \t \t \t \t \t At least one RecHit inside the range, Predicted="<<stripPredicted<<" minres="<<minres<<"cm range="<<rangestrips<<"strips stripw="<<stripw<<"cm"<<std::endl;
			  if(debug) std::cout<<"MB4  \t \t \t \t \t Norm of Cosine Directors="<<dx3*dx3+dy3*dy3+dz3*dz3<<"~1?"<<std::endl;
		   
			  //-----RESIDUALS----------
			  if(inves){
			    float cosal = dx/sqrt(dx*dx+dz*dz);
			    if(debug) std::cout<<"MB4 \t \t \t \t \t Angle="<<acos(cosal)*180/3.1415926<<" degree"<<std::endl;
			    if(debug) std::cout<<"MB4 \t \t \t \t \t Filling the Residuals Histogram for globals with "<<minres<<"And the angular incidence with Cos Theta="<<-1*dz<<std::endl;
			    assert(rollId.station()==4);
			    if(cluSize==1*dupli){ hGlobalResClu1La6->Fill(minres); if(minres<0.005*stripw) AngClu1La6->Fill(acos(cosal)); if(fabs(minres)<stripw*0.5) DistBorderClu1La6->Fill(minres/stripw+0.5);}
			    else if(cluSize==2*dupli){ hGlobalResClu2La6->Fill(minres); if(minres<0.005*stripw) AngClu2La6->Fill(acos(cosal));}
			    else if(cluSize==3*dupli){ hGlobalResClu3La6->Fill(minres); if(minres<0.005*stripw) AngClu3La6->Fill(acos(cosal));}
			  }
			  //--------------------------------
			  
			  sprintf(meIdRPC,"RPCDataOccupancyFromDT_%s",detUnitLabel);
			  if(debug) std::cout<<"MB4 \t \t \t \t \t \t COINCIDENCE!!! Event="<<iEvent.id()<<"Filling RPC Data Occupancy for "<<meIdRPC<<" with "<<stripPredicted<<std::endl; 
			  meMap[meIdRPC]->Fill(stripPredicted);
			}
			else{
			  RPCGeomServ rpcsrv(rollasociated->id());
			  std::string nameRoll = rpcsrv.name();
			  if(debug) std::cout<<"MB4 \t \t \t \t \t \t A roll was ineficient in event"<<iEvent.id().event()<<std::endl;
			  if(debug) ofrej<<"MB4 \t Wh "<<dtWheel
					 <<"\t St "<<dtStation
					 <<"\t Se "<<dtSector
					 <<"\t Roll "<<nameRoll
					 <<"\t Event "
					 <<iEvent.id().event()
					 <<"\t Run "
					 <<iEvent.id().run()
					 <<std::endl;
			}
		      }else{
			if(debug) std::cout<<"MB4 \t \t \t \t No the prediction is outside of this roll"<<std::endl;
		      }
		    }//Condition for the right match
		    else{
		      if(debug) std::cout<<"MB4 \t \t \t No, Exrtrapolation too long!, canceled"<<std::endl;
		    }
		  }//loop over all the rollsasociated
		}else{
		  compatiblesegments=false;
		  if(debug) std::cout<<"MB4 \t \t \t \t I found segments in MB4 and MB3 adjacent or same wheel and sector but not compatibles Diferent Directions"<<std::endl;
		}
	      }else{//if dtid3.station()==3&&dtid3.sector()==DTId.sector()&&dtid3.wheel()==DTId.wheel()&&segMB3->dim()==4
		if(debug) std::cout<<"MB4 \t \t \t No the same station or same wheel or segment dim in mb3 not 4D"<<std::endl;
	      }
	    }//loop over all the segments looking for one in MB3 
	  }else{
	    if(debug) std::cout<<"MB4 \t \t \t Is NOT a 2D Segment"<<std::endl;
	  }
	}else{
	  if(debug) std::cout<<"MB4 \t \t \t \t There is not just one segment or is not in station 4"<<std::endl;
	}//De aca para abajo esta en dtpart.inl
      }
    }else{
      if(debug) std::cout<<"MB4 \t This event doesn't have 4D Segment"<<std::endl;
    }
  }

  if(inclcsc){
    if(debug) std::cout <<"\t Getting the CSC Segments"<<std::endl;
    edm::Handle<CSCSegmentCollection> allCSCSegments;
    iEvent.getByLabel(cscSegments, allCSCSegments);
    
    if(allCSCSegments->size()>0){
      statistics->Fill(18);

      if(debug) std::cout<<"CSC \t Number of CSC Segments in this event = "<<allCSCSegments->size()<<std::endl;
      
      std::map<CSCDetId,int> CSCSegmentsCounter;
      CSCSegmentCollection::const_iterator segment;
      
      int segmentsInThisEventInTheEndcap=0;
      
      for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
	CSCSegmentsCounter[segment->cscDetId()]++;
	segmentsInThisEventInTheEndcap++;
      }    
      
      statistics->Fill(allCSCSegments->size()+18);
      
      
      if(debug) std::cout<<"CSC \t loop over all the CSCSegments "<<std::endl;
      for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
	CSCDetId CSCId = segment->cscDetId();
	
	if(debug) std::cout<<"CSC \t \t This Segment is in Chamber id: "<<CSCId<<std::endl;
	if(debug) std::cout<<"CSC \t \t Number of segments in this CSC = "<<CSCSegmentsCounter[CSCId]<<std::endl;
	if(debug) std::cout<<"CSC \t \t Is the only one in this CSC? is not ind the ring 1 or station 4? Are there more than 2 segments in the event?"<<std::endl;

    	if(CSCSegmentsCounter[CSCId]==1 && CSCId.station()!=4 && CSCId.ring()!=1 && allCSCSegments->size()>=2){
	  if(debug) std::cout<<"CSC \t \t yes"<<std::endl;
    	  int cscEndCap = CSCId.endcap();
	  int cscStation = CSCId.station();
	  int cscRing = CSCId.ring();
	  int cscChamber = CSCId.chamber();
	  int rpcRegion = 1; if(cscEndCap==2) rpcRegion= -1;//Relacion entre las endcaps
	  int rpcRing = cscRing;
	  if(cscRing==4)rpcRing =1;
	  int rpcStation = cscStation;
	  int rpcSegment = 0;
	
	  if(cscStation!=1&&cscRing==1){//las de 18 CSC
	    rpcSegment = CSCId.chamber();
	  }
	  else{//las de 36 CSC
	    rpcSegment = (CSCId.chamber()==1) ? 36 : CSCId.chamber()-1;
	  }
     
	  LocalPoint segmentPosition= segment->localPosition();
	  LocalVector segmentDirection=segment->localDirection();
	  float dz=segmentDirection.z();
	
	  if(debug) std::cout<<"CSC \t \t Is a good Segment? dim = 4, 4 <= nRecHits <= 10 Incident angle int range 45 < "<<acos(dz)*180/3.1415926<<" < 135? "<<std::endl;

	  if(segment->dimension()==4 && (segment->nRecHits()<=10 && segment->nRecHits()>=4)&& acos(dz)*180/3.1415926 > 45. && acos(dz)*180/3.1415926 < 160. ){ 
	    
	    //&& segment->chi2()< ??)Add 3 segmentes in the endcaps???

	    if(debug) std::cout<<"CSC \t \t yes"<<std::endl;
	    if(debug) std::cout<<"CSC \t \t CSC Segment Dimension "<<segment->dimension()<<std::endl; 
	    
	    float Xo=segmentPosition.x();
	    float Yo=segmentPosition.y();
	    float Zo=segmentPosition.z();
	    float dx=segmentDirection.x();
	    float dy=segmentDirection.y();
	    float dz=segmentDirection.z();

	    
	    if(debug) std::cout<<"CSC \t \t Getting chamber from Geometry"<<std::endl;
	    const CSCChamber* TheChamber=cscGeo->chamber(CSCId); 
	    if(debug) std::cout<<"CSC \t \t Getting ID from Chamber"<<std::endl;
	    const CSCDetId TheId=TheChamber->id();
	    if(debug) std::cout<<"CSC \t \t Printing The Id"<<TheId<<std::endl;
	    std::set<RPCDetId> rollsForThisCSC = rollstoreCSC[CSCStationIndex(rpcRegion,rpcStation,rpcRing,rpcSegment)];
	    if(debug) std::cout<<"CSC \t \t Number of rolls for this CSC = "<<rollsForThisCSC.size()<<std::endl;

	    if(debug) std::cout<<"CSC \t \t Loop over all the rolls asociated to this CSC"<<std::endl;	    

	    if(rpcRing!=1&&rpcStation!=4){
	  
	      if(rollsForThisCSC.size()==0){
		if(debug) std::cout<<"CSC Fail for CSCId="<<TheId<<" rpcRegion="<<rpcRegion<<" rpcStation="<<rpcStation<<" rpcRing="<<rpcRing<<" rpcSegment="<<rpcSegment<<std::endl;
	      }
	      
	      assert(rollsForThisCSC.size()>=1);

	      //Loop over all the rolls
	      for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisCSC.begin();iteraRoll != rollsForThisCSC.end(); iteraRoll++){
		const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
		RPCDetId rpcId = rollasociated->id();
		
		if(debug) std::cout<<"CSC \t \t \t We are in the roll getting the surface"<<rpcId<<std::endl;
		const BoundPlane & RPCSurface = rollasociated->surface(); 

		if(debug) std::cout<<"CSC \t \t \t RollID: "<<rpcId<<std::endl;
		
		if(debug) std::cout<<"CSC \t \t \t Doing the extrapolation to this roll"<<std::endl;
		if(debug) std::cout<<"CSC \t \t \t CSC Segment Direction in CSCLocal "<<segmentDirection<<std::endl;
		if(debug) std::cout<<"CSC \t \t \t CSC Segment Point in CSCLocal "<<segmentPosition<<std::endl;  
		
		GlobalPoint CenterPointRollGlobal = RPCSurface.toGlobal(LocalPoint(0,0,0));
		if(debug) std::cout<<"CSC \t \t \t Center (0,0,0) of the Roll in Global"<<CenterPointRollGlobal<<std::endl;
		GlobalPoint CenterPointCSCGlobal = TheChamber->toGlobal(LocalPoint(0,0,0));
		if(debug) std::cout<<"CSC \t \t \t Center (0,0,0) of the CSC in Global"<<CenterPointCSCGlobal<<std::endl;
		GlobalPoint segmentPositionInGlobal=TheChamber->toGlobal(segmentPosition); //new way to convert to global
		if(debug) std::cout<<"CSC \t \t \t Segment Position in Global"<<segmentPositionInGlobal<<std::endl;
		LocalPoint CenterRollinCSCFrame = TheChamber->toLocal(CenterPointRollGlobal);

		if(debug){//to check CSC RPC phi relation!
		  float rpcphi=0;
		  float cscphi=0;
		  
		  (CenterPointRollGlobal.barePhi()<0)? 
		    rpcphi = 2*3.141592+CenterPointRollGlobal.barePhi():rpcphi=CenterPointRollGlobal.barePhi();
		  
		  (CenterPointCSCGlobal.barePhi()<0)? 
		    cscphi = 2*3.1415926536+CenterPointCSCGlobal.barePhi():cscphi=CenterPointCSCGlobal.barePhi();

		  float df=fabs(cscphi-rpcphi); 
		  float dr=fabs(CenterPointRollGlobal.perp()-CenterPointCSCGlobal.perp());
		  float diffz=CenterPointRollGlobal.z()-CenterPointCSCGlobal.z();
		  float dfg=df*180./3.14159265;

		  if(debug) std::cout<<"CSC \t \t \t z of RPC="<<CenterPointRollGlobal.z()<<"z of CSC"<<CenterPointCSCGlobal.z()<<" dfg="<<dfg<<std::endl;


		  RPCGeomServ rpcsrv(rpcId);

		  
		  if(dr>200.||fabs(dz)>55.||dfg>1.){ 
		    //if(rpcRegion==1&&dfg>1.&&dr>100.){  
		    if (debug) std::cout
		      <<"\t \t \t CSC Station= "<<CSCId.station()
		      <<" Ring= "<<CSCId.ring()
		      <<" Chamber= "<<CSCId.chamber()
		      <<" cscphi="<<cscphi*180/3.14159265
		      <<"\t RPC Station= "<<rpcId.station()
		      <<" ring= "<<rpcId.ring()
		      <<" segment =-> "<<mySegment(rpcId)
		      <<" rollphi="<<rpcphi*180/3.14159265
		      <<"\t dfg="<<dfg
		      <<" dz="<<diffz
		      <<" dr="<<dr
		      <<std::endl;
		    
		  }
		}



	    
		float D=CenterRollinCSCFrame.z();
	  	  
		float X=Xo+dx*D/dz;
		float Y=Yo+dy*D/dz;
		float Z=D;

		const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(rollasociated->topology()));
		LocalPoint xmin = top_->localPosition(0.);
		if(debug) std::cout<<"CSC \t \t \t xmin of this  Roll "<<xmin<<"cm"<<std::endl;
		LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
		if(debug) std::cout<<"CSC \t \t \t xmax of this  Roll "<<xmax<<"cm"<<std::endl;
		float rsize = fabs( xmax.x()-xmin.x() );
		if(debug) std::cout<<"CSC \t \t \t Roll Size "<<rsize<<"cm"<<std::endl;
		float stripl = top_->stripLength();
		float stripw = top_->pitch();

		if(debug) std::cout<<"CSC \t \t \t Strip Lenght "<<stripl<<"cm"<<std::endl;
		if(debug) std::cout<<"CSC \t \t \t Strip Width "<<stripw<<"cm"<<std::endl;

		if(debug) std::cout<<"CSC \t \t \t X Predicted in CSCLocal= "<<X<<"cm"<<std::endl;
		if(debug) std::cout<<"CSC \t \t \t Y Predicted in CSCLocal= "<<Y<<"cm"<<std::endl;
		if(debug) std::cout<<"CSC \t \t \t Z Predicted in CSCLocal= "<<Z<<"cm"<<std::endl;
	  
		float extrapolatedDistance = sqrt((X-Xo)*(X-Xo)+(Y-Yo)*(Y-Yo)+(Z-Zo)*(Z-Zo));

		if(debug) std::cout<<"CSC \t \t \t Is the distance of extrapolation less than MaxD? ="<<extrapolatedDistance<<"cm"<<"MaxD="<<MaxD<<"cm"<<std::endl;
	  
		if(extrapolatedDistance<=MaxD){ 

		  if(debug) std::cout<<"CSC \t \t \t yes"<<std::endl;

		  GlobalPoint GlobalPointExtrapolated=TheChamber->toGlobal(LocalPoint(X,Y,Z));
		  if(debug) std::cout<<"CSC \t \t \t Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;

	      
		  LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);

		  if(debug) std::cout<<"CSC \t \t \t Point Extrapolated in RPCLocal"<<PointExtrapolatedRPCFrame<< std::endl;
		  if(debug) std::cout<<"CSC \t \t \t Corner of the Roll = ("<<rsize*0.5<<","<<stripl*0.5<<")"<<std::endl;
		  if(debug) std::cout<<"CSC \t \t \t Info About the Point Extrapolated in X Abs ("<<fabs(PointExtrapolatedRPCFrame.x())<<","
				     <<fabs(PointExtrapolatedRPCFrame.y())<<","<<fabs(PointExtrapolatedRPCFrame.z())<<")"<<std::endl;
		  if(debug) std::cout<<"CSC \t \t \t dz="
				     <<fabs(PointExtrapolatedRPCFrame.z())<<" dx="
				     <<fabs(PointExtrapolatedRPCFrame.x())<<" dy="
				     <<fabs(PointExtrapolatedRPCFrame.y())<<std::endl;
		  
		  if(debug) std::cout<<"CSC \t \t \t Does the extrapolation go inside this roll????"<<std::endl;

		  if(fabs(PointExtrapolatedRPCFrame.z()) < 10. && 
		     fabs(PointExtrapolatedRPCFrame.x()) < rsize*0.5 && 
		     fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
		    
		    if(debug) std::cout<<"CSC \t \t \t \t yes"<<std::endl;

		    RPCDetId  rollId = rollasociated->id();
		    
		    RPCGeomServ rpcsrv(rollId);
		    std::string nameRoll = rpcsrv.name();
		    if(debug) std::cout<<"CSC \t \t \t \t The RPCName is "<<nameRoll<<std::endl;

		    const float stripPredicted = 
		      rollasociated->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); 

		    if(debug) std::cout<<"CSC  \t \t \t \t \t Candidate"<<rollId<<" "<<"(from CSC Segment) STRIP---> "<<stripPredicted<< std::endl;
		    //--------- HISTOGRAM STRIP PREDICTED FROM CSC  -------------------
		    
		    char detUnitLabel[128];
		    sprintf(detUnitLabel ,"%s",nameRoll.c_str());
		    sprintf(layerLabel ,"%s",nameRoll.c_str());
		    
		    std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
		    
		    if(debug) std::cout<<"CSC \t \t \t \t Filling Expected"<<std::endl;
		    sprintf(meIdCSC,"ExpectedOccupancyFromCSC_%s",detUnitLabel);
		    meMap[meIdCSC]->Fill(stripPredicted);
		    //--------------------------------------------------------------------
	    		
		    
		    //-------RecHitPart Just For Residual--------
		    int cluSize = 0;
		    int countRecHits = 0;
		    float minres = 3000.;
		    
		    if(debug) std::cout<<"CSC  \t \t \t \t \t Getting RecHits in Roll Asociated"<<std::endl;
		    typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
		    rangeRecHits recHitCollection =  rpcHits->get(rollasociated->id());
		    RPCRecHitCollection::const_iterator recHit;

		    for (recHit = recHitCollection.first; recHit != recHitCollection.second ; recHit++) {
		      countRecHits++;
		      LocalPoint recHitPos=recHit->localPosition();
		      float res=PointExtrapolatedRPCFrame.x()- recHitPos.x();
		      if(debug) std::cout<<"CSC  \t \t \t \t \t \t Found Rec Hit at "<<res<<"cm of the prediction."<<std::endl;
		      if(fabs(res)<fabs(minres)){
			minres=res;
			cluSize = recHit->clusterSize();
			if(debug) std::cout<<"CSC  \t \t \t \t \t \t \t New Min Res "<<res<<"cm."<<std::endl;
		      }
		    }
		    
		    bool anycoincidence = false;
		    
		    if(countRecHits==0){
		      if(debug) std::cout <<"CSC \t \t \t \t \t THIS ROLL DOESN'T HAVE ANY RECHIT"<<std::endl;
		    }else{  
		      assert(minres!=3000); 
		      
		      if(debug) std::cout<<"CSC \t \t \t \t \t PointExtrapolatedRPCFrame.x="<<PointExtrapolatedRPCFrame.x()<<" Minimal Residual"<<minres<<std::endl;
		      if(debug) std::cout<<"CSC  \t \t \t \t \t Minimal Residual less than stripw*rangestrips? minres="<<minres<<" range="<<rangestrips<<" stripw="<<stripw<<" cluSize"<<cluSize<<" <=compare minres with"<<(rangestrips+cluSize*0.5)*stripw<<std::endl;
		      if(fabs(minres)<=(rangestrips+cluSize*0.5)*stripw){
			if(debug) std::cout<<"CSC  \t \t \t \t \t \t True!"<<std::endl;
			anycoincidence=true;
		      }
		    }
		    if(anycoincidence){
		      if(debug) std::cout<<"CSC  \t \t \t \t \t At least one RecHit inside the range, Predicted="<<stripPredicted<<" minres="<<minres<<"cm range="<<rangestrips<<"strips stripw="<<stripw<<"cm"<<std::endl;
		      if(debug) std::cout<<"CSC  \t \t \t \t \t Norm of Cosine Directors="<<dx*dx+dy*dy+dz*dz<<"~1?"<<std::endl;

		      //----RESIDUALS----
		      if(inves){
			if(rollId.ring()==2&&rollId.roll()==1){if(cluSize==1*dupli) hGlobalResClu1R2A->Fill(minres); if(cluSize==2*dupli) hGlobalResClu2R2A->Fill(minres); if(cluSize==3*dupli) hGlobalResClu3R2A->Fill(minres);}
			if(rollId.ring()==2&&rollId.roll()==2){if(cluSize==1*dupli) hGlobalResClu1R2B->Fill(minres); if(cluSize==2*dupli) hGlobalResClu2R2B->Fill(minres); if(cluSize==3*dupli) hGlobalResClu3R2B->Fill(minres);}
			if(rollId.ring()==2&&rollId.roll()==3){if(cluSize==1*dupli) hGlobalResClu1R2C->Fill(minres); if(cluSize==2*dupli) hGlobalResClu2R2C->Fill(minres); if(cluSize==3*dupli) hGlobalResClu3R2C->Fill(minres);}
			if(rollId.ring()==3&&rollId.roll()==1){if(cluSize==1*dupli) hGlobalResClu1R3A->Fill(minres); if(cluSize==2*dupli) hGlobalResClu2R3A->Fill(minres); if(cluSize==3*dupli) hGlobalResClu3R3A->Fill(minres);}
			if(rollId.ring()==3&&rollId.roll()==2){if(cluSize==1*dupli) hGlobalResClu1R3B->Fill(minres); if(cluSize==2*dupli) hGlobalResClu2R3B->Fill(minres); if(cluSize==3*dupli) hGlobalResClu3R3B->Fill(minres);}
			if(rollId.ring()==3&&rollId.roll()==3){if(cluSize==1*dupli) hGlobalResClu1R3C->Fill(minres); if(cluSize==2*dupli) hGlobalResClu2R3C->Fill(minres); if(cluSize==3*dupli) hGlobalResClu3R3C->Fill(minres);}
		      }
		      //------------------------
		      sprintf(meIdRPC,"RPCDataOccupancyFromCSC_%s",detUnitLabel);
		      if(debug) std::cout <<"CSC \t \t \t \t \t \t COINCEDENCE!!! Event="<<iEvent.id()<<"Filling RPC Data Occupancy for "<<meIdRPC<<" with "<<stripPredicted<<std::endl;
		      meMap[meIdRPC]->Fill(stripPredicted);
		    }
		    else{
		      RPCGeomServ rpcsrv(rollasociated->id());
		      std::string nameRoll = rpcsrv.name();
		      if(debug) std::cout<<"CSC \t \t \t \t \t \t A roll was ineficient in event"<<iEvent.id().event()<<std::endl;
		      if(debug) ofrej<<"CSC \t EndCap "<<rpcRegion
				     <<"\t cscStation "<<cscStation
				     <<"\t cscRing "<<cscRing			   
				     <<"\t cscChamber "<<cscChamber
				     <<"\t Roll "<<nameRoll
				     <<"\t Event "<<iEvent.id().event()
				     <<"\t CSCId "<<CSCId
				     <<"\t Event "	
				     <<iEvent.id().event()
				     <<"\t Run "
				     <<iEvent.id().run()
				     <<std::endl;
		    }
		  }else{
		    if(debug) std::cout<<"CSC \t \t \t \t No the prediction is outside of this roll"<<std::endl;
		  }//Condition for the right match
		}else{//if extrapolation distance D is not too long
		  if(debug) std::cout<<"CSC \t \t \t No, Exrtrapolation too long!, canceled"<<std::endl;
		}//D so big
	      }//loop over the rolls asociated 
	    }//Condition over the startup geometry!!!!
	  }//Is the segment 4D?
	}else{
	  if(debug) std::cout<<"CSC \t \t More than one segment in this chamber, or we are in Station Ring 1 or in Station 4"<<std::endl;
	}
      }
    }else{
      if(debug) std::cout<<"CSC This Event doesn't have any CSCSegment"<<std::endl;
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
  dbe =0;
}
