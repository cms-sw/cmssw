/***************************************
Author: 
Camilo Carrillo
Universidad de los Andes Bogota Colombia
camilo.carrilloATcern.ch
****************************************/

#include "DQM/RPCMonitorDigi/interface/utils.h"
#include "DQM/RPCMonitorClient/interface/RPCEfficiencySecond.h"
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

RPCEfficiencySecond::RPCEfficiencySecond(const edm::ParameterSet& iConfig){
  SaveFile  = iConfig.getUntrackedParameter<bool>("SaveFile", false); 
  NameFile  = iConfig.getUntrackedParameter<std::string>("NameFile","RPCEfficiency.root"); 
  folderPath  = iConfig.getUntrackedParameter<std::string>("folderPath","RPC/RPCEfficiency/"); 
  debug = iConfig.getUntrackedParameter<bool>("debug",false); 
  barrel = iConfig.getUntrackedParameter<bool>("barrel"); 
  endcap = iConfig.getUntrackedParameter<bool>("endcap"); 
}
 
RPCEfficiencySecond::~RPCEfficiencySecond(){}

void RPCEfficiencySecond::beginRun(const edm::Run&, const edm::EventSetup& iSetup){
  
  dbe = edm::Service<DQMStore>().operator->();

  //Barrel 
  
  dbe->setCurrentFolder(folderPath+"Wheel_-2");
  EffDistroWm2=dbe->book1D("EffDistroWheel_-2","Efficiency Distribution for Wheel -2 ",20,0.5,100.5);
  EffGlobWm2=dbe->book1D("GlobEfficiencyWheel_-2","Efficiency Wheel -2 ",206,0.5,206.5);
  
  dbe->setCurrentFolder(folderPath+"Wheel_-1");
  EffDistroWm1=dbe->book1D("EffDistroWheel_-1","Efficiency Distribution for Wheel -1 ",20,0.5,100.5);
  EffGlobWm1= dbe->book1D("GlobEfficiencyWheel_-1","Efficiency Wheel -1",206,0.5,206.5);


  dbe->setCurrentFolder(folderPath+"Wheel_0");
  EffDistroW0=dbe->book1D("EffDistroWheel_0","Efficiency Distribution for Wheel 0 ",20,0.5,100.5);
  EffGlobW0 = dbe->book1D("GlobEfficiencyWheel_0","Efficiency Wheel 0",206,0.5,206.5);


  dbe->setCurrentFolder(folderPath+"Wheel_1");
  EffDistroW1=dbe->book1D("EffDistroWheel_1","Efficiency Distribution for Wheel 1 ",20,0.5,100.5);
  EffGlobW1 = dbe->book1D("GlobEfficiencyWheel_1","Efficiency Wheel 1",206,0.5,206.5);


  dbe->setCurrentFolder(folderPath+"Wheel_2");
  EffDistroW2=dbe->book1D("EffDistroWheel_2","Efficiency Distribution for Wheel 2 ",20,0.5,100.5);
  EffGlobW2 = dbe->book1D("GlobEfficiencyWheel_2","Efficiency Wheel 2",206,0.5,206.5);


  //EndCap
  dbe->setCurrentFolder(folderPath+"Disk_3");
  EffDistroD3=dbe->book1D("EffDistroDisk_3","Efficiency Distribution Disk 3 ",20,0.5,100.5);
  EffGlobD3 = dbe->book1D("GlobEfficiencyDisk_3","Efficiency Disk 3",218,0.5,218.5);

  dbe->setCurrentFolder(folderPath+"Disk_2");
  EffDistroD2=dbe->book1D("EffDistroDisk_2","Efficiency Distribution Disk 2 ",20,0.5,100.5);
  EffGlobD2 = dbe->book1D("GlobEfficiencyDisk_2","Efficiency Disk 2",218,0.5,218.5);


  dbe->setCurrentFolder(folderPath+"Disk_1");
  EffDistroD1=dbe->book1D("EffDistroDisk_1","Efficiency Distribution Disk 1 ",20,0.5,100.5);
  EffGlobD1 = dbe->book1D("GlobEfficiencyDisk_1","Efficiency Disk 1",218,0.5,218.5);


  dbe->setCurrentFolder(folderPath+"Disk_-1");
  EffDistroDm1=dbe->book1D("EffDistroDisk_m1","Efficiency Distribution Disk - 1 ",20,0.5,100.5);
  EffGlobDm1 = dbe->book1D("GlobEfficiencyDisk_m1","Efficiency Disk -1",218,0.5,218.5);

  dbe->setCurrentFolder(folderPath+"Disk_-2");
  EffDistroDm2=dbe->book1D("EffDistroDisk_m2","Efficiency Distribution Disk - 2 ",20,0.5,100.5);
  EffGlobDm2 = dbe->book1D("GlobEfficiencyDisk_m2","Efficiency Disk -2",218,0.5,218.5);


  dbe->setCurrentFolder(folderPath+"Disk_-3");
  EffDistroDm3=dbe->book1D("EffDistroDisk_m3","Efficiency Distribution Disk - 3 ",20,0.5,100.5);
  EffGlobDm3 = dbe->book1D("GlobEfficiencyDisk_m3","Efficiency Disk -3",218,0.5,218.5);


  //Summary Histograms
  dbe->setCurrentFolder(folderPath);
  std::string os;
  os="Efficiency_Roll_vs_Sector_Wheel_-2";                               
  Wheelm2Summary = dbe->book2D(os, os, 12, 0.5,12.5, 21, 0.5, 21.5);
  os="Efficiency_Roll_vs_Sector_Wheel_-1";                                      
  Wheelm1Summary = dbe->book2D(os, os, 12, 0.5,12.5, 21, 0.5, 21.5);
  os="Efficiency_Roll_vs_Sector_Wheel_0";                                      
  Wheel0Summary = dbe->book2D(os, os, 12, 0.5,12.5, 21, 0.5, 21.5);
  os="Efficiency_Roll_vs_Sector_Wheel_+1";                                      
  Wheel1Summary = dbe->book2D(os, os, 12, 0.5,12.5, 21, 0.5, 21.5);
  os="Efficiency_Roll_vs_Sector_Wheel_+2";                                      
  Wheel2Summary = dbe->book2D(os, os, 12, 0.5,12.5, 21, 0.5, 21.5);

  rpcdqm::utils rpcUtils;
  rpcUtils.labelXAxisSector( Wheel2Summary );
  rpcUtils.labelYAxisRoll( Wheel2Summary, 0, 2, true);

  rpcUtils.labelXAxisSector( Wheel1Summary );
  rpcUtils.labelYAxisRoll( Wheel1Summary, 0, 1, true);

  rpcUtils.labelXAxisSector( Wheel0Summary );
  rpcUtils.labelYAxisRoll( Wheel0Summary, 0, 0, true);

  rpcUtils.labelXAxisSector( Wheelm1Summary );
  rpcUtils.labelYAxisRoll( Wheelm1Summary, 0, -1, true);

  rpcUtils.labelXAxisSector( Wheelm2Summary );
  rpcUtils.labelYAxisRoll( Wheelm2Summary, 0, -2, true);

  os="Efficiency_Roll_vs_Segment_Disk_-3";
  Diskm3Summary = dbe->book2D(os,os,36,0.5,36.5,6,0.5,6.5);
  os="Efficiency_Roll_vs_Segment_Disk_-2";
  Diskm2Summary = dbe->book2D(os,os,36,0.5,36.5,6,0.5,6.5);
  os="Efficiency_Roll_vs_Segment_Disk_-1";
  Diskm1Summary = dbe->book2D(os,os,36,0.5,36.5,6,0.5,6.5);
  os="Efficiency_Roll_vs_Segment_Disk_1";
  Disk1Summary = dbe->book2D(os,os,36,0.5,36.5,6,0.5,6.5);
  os="Efficiency_Roll_vs_Segment_Disk_2";
  Disk2Summary = dbe->book2D(os,os,36,0.5,36.5,6,0.5,6.5);
  os="Efficiency_Roll_vs_Segment_Disk_3";
  Disk3Summary = dbe->book2D(os,os,36,0.5,36.5,6,0.5,6.5);

  rpcUtils.labelXAxisSegment(Diskm3Summary);
  rpcUtils.labelYAxisRing(Diskm3Summary, 2, true);

  rpcUtils.labelXAxisSegment(Diskm2Summary);
  rpcUtils.labelYAxisRing(Diskm2Summary, 2, true);

  rpcUtils.labelXAxisSegment(Diskm1Summary);
  rpcUtils.labelYAxisRing(Diskm1Summary, 2, true);

  rpcUtils.labelXAxisSegment(Disk1Summary);
  rpcUtils.labelYAxisRing(Disk1Summary, 2, true);

  rpcUtils.labelXAxisSegment(Disk2Summary);
  rpcUtils.labelYAxisRing(Disk2Summary, 2, true);

  rpcUtils.labelXAxisSegment(Disk3Summary);
  rpcUtils.labelYAxisRing(Disk3Summary, 2, true);
  
  //Azimutal Histograms

  dbe->setCurrentFolder(folderPath+"Azimutal/");
  sectorEffWm2= dbe->book1D("AzimutalDistroWm2","Efficiency per Sector Wheel -2",12,0.5,12.5);
  sectorEffWm1= dbe->book1D("AzimutalDistroWm1","Efficiency per Sector Wheel -1",12,0.5,12.5);
  sectorEffW0= dbe->book1D("AzimutalDistroW0","Efficiency per Sector Wheel 0",12,0.5,12.5);
  sectorEffW1= dbe->book1D("AzimutalDistroW1","Efficiency per Sector Wheel 1",12,0.5,12.5);
  sectorEffW2= dbe->book1D("AzimutalDistroW2","Efficiency per Sector Wheel 2",12,0.5,12.5);

  OcsectorEffWm2= dbe->book1D("AzimutalDistroWm2Oc","Occupancy per Sector Wheel -2",12,0.5,12.5);
  OcsectorEffWm1= dbe->book1D("AzimutalDistroWm1Oc","Occupancy per Sector Wheel -1",12,0.5,12.5);
  OcsectorEffW0= dbe->book1D("AzimutalDistroW0Oc","Ocuppancy per Sector Wheel 0",12,0.5,12.5);
  OcsectorEffW1= dbe->book1D("AzimutalDistroW1Oc","Ocuppancy per Sector Wheel 1",12,0.5,12.5);
  OcsectorEffW2= dbe->book1D("AzimutalDistroW2Oc","Ocupancy per Sector Wheel 2",12,0.5,12.5);

  ExsectorEffWm2= dbe->book1D("AzimutalDistroWm2Ex","Expected per Sector Wheel -2",12,0.5,12.5);
  ExsectorEffWm1= dbe->book1D("AzimutalDistroWm1Ex","Expected per Sector Wheel -1",12,0.5,12.5);
  ExsectorEffW0= dbe->book1D("AzimutalDistroW0Ex","Expected per Sector Wheel 0",12,0.5,12.5);
  ExsectorEffW1= dbe->book1D("AzimutalDistroW1Ex","Expected per Sector Wheel 1",12,0.5,12.5);
  ExsectorEffW2= dbe->book1D("AzimutalDistroW2Ex","Expected per Sector Wheel 2",12,0.5,12.5);
  
  GregD1R2= dbe->book1D("GregDistroD1R2","Efficiency for Station 1 Ring 2",36,0.5,36.5);
  GregD1R3= dbe->book1D("GregDistroD1R3","Efficiency for Station 1 Ring 3",36,0.5,36.5);
  GregD2R2= dbe->book1D("GregDistroD2R2","Efficiency for Station 2 Ring 2",36,0.5,36.5);
  GregD2R3= dbe->book1D("GregDistroD2R3","Efficiency for Station 2 Ring 3",36,0.5,36.5);
  GregD3R2= dbe->book1D("GregDistroD3R2","Efficiency for Station 3 Ring 2",36,0.5,36.5);
  GregD3R3= dbe->book1D("GregDistroD3R3","Efficiency for Station 3 Ring 3",36,0.5,36.5);
  GregDm1R2= dbe->book1D("GregDistroDm1R2","Efficiency for Station -1 Ring 2",36,0.5,36.5);
  GregDm1R3= dbe->book1D("GregDistroDm1R3","Efficiency for Station -1 Ring 3",36,0.5,36.5);
  GregDm2R2= dbe->book1D("GregDistroDm2R2","Efficiency for Station -2 Ring 2",36,0.5,36.5);
  GregDm2R3= dbe->book1D("GregDistroDm2R3","Efficiency for Station -2 Ring 3",36,0.5,36.5);
  GregDm3R2= dbe->book1D("GregDistroDm3R2","Efficiency for Station -3 Ring 2",36,0.5,36.5);
  GregDm3R3= dbe->book1D("GregDistroDm3R3","Efficiency for Station -3 Ring 3",36,0.5,36.5);

  OcGregD1R2= dbe->book1D("OcGregDistroD1R2","Occupancy Distribution for Station 1 Ring 2",36,0.5,36.5);
  OcGregD1R3= dbe->book1D("OcGregDistroD1R3","Occupancy Distribution for Station 1 Ring 3",36,0.5,36.5);
  OcGregD2R2= dbe->book1D("OcGregDistroD2R2","Occupancy Distribution for Station 2 Ring 2",36,0.5,36.5);
  OcGregD2R3= dbe->book1D("OcGregDistroD2R3","Occupancy Distribution for Station 2 Ring 3",36,0.5,36.5);
  OcGregD3R2= dbe->book1D("OcGregDistroD3R2","Occupancy Distribution for Station 3 Ring 2",36,0.5,36.5);
  OcGregD3R3= dbe->book1D("OcGregDistroD3R3","Occupancy Distribution for Station 3 Ring 3",36,0.5,36.5);
  OcGregDm1R2= dbe->book1D("OcGregDistroDm1R2","Occupancy Distribution for Station -1 Ring 2",36,0.5,36.5);
  OcGregDm1R3= dbe->book1D("OcGregDistroDm1R3","Occupancy Distribution for Station -1 Ring 3",36,0.5,36.5);
  OcGregDm2R2= dbe->book1D("OcGregDistroDm2R2","Occupancy Distribution for Station -2 Ring 2",36,0.5,36.5);
  OcGregDm2R3= dbe->book1D("OcGregDistroDm2R3","Occupancy Distribution for Station -2 Ring 3",36,0.5,36.5);
  OcGregDm3R2= dbe->book1D("OcGregDistroDm3R2","Occupancy Distribution for Station -3 Ring 2",36,0.5,36.5);
  OcGregDm3R3= dbe->book1D("OcGregDistroDm3R3","Occupancy Distribution for Station -3 Ring 3",36,0.5,36.5);

  ExGregD1R2= dbe->book1D("ExGregDistroD1R2","Expected Distribution for Station 1 Ring 2",36,0.5,36.5);
  ExGregD1R3= dbe->book1D("ExGregDistroD1R3","Expected Distribution for Station 1 Ring 3",36,0.5,36.5);
  ExGregD2R2= dbe->book1D("ExGregDistroD2R2","Expected Distribution for Station 2 Ring 2",36,0.5,36.5);
  ExGregD2R3= dbe->book1D("ExGregDistroD2R3","Expected Distribution for Station 2 Ring 3",36,0.5,36.5);
  ExGregD3R2= dbe->book1D("ExGregDistroD3R2","Expected Distribution for Station 3 Ring 2",36,0.5,36.5);
  ExGregD3R3= dbe->book1D("ExGregDistroD3R3","Expected Distribution for Station 3 Ring 3",36,0.5,36.5);
  ExGregDm1R2= dbe->book1D("ExGregDistroDm1R2","Expected Distribution for Station -1 Ring 2",36,0.5,36.5);
  ExGregDm1R3= dbe->book1D("ExGregDistroDm1R3","Expected Distribution for Station -1 Ring 3",36,0.5,36.5);
  ExGregDm2R2= dbe->book1D("ExGregDistroDm2R2","Expected Distribution for Station -2 Ring 2",36,0.5,36.5);
  ExGregDm2R3= dbe->book1D("ExGregDistroDm2R3","Expected Distribution for Station -2 Ring 3",36,0.5,36.5);
  ExGregDm3R2= dbe->book1D("ExGregDistroDm3R2","Expected Distribution for Station -3 Ring 2",36,0.5,36.5);
  ExGregDm3R3= dbe->book1D("ExGregDistroDm3R3","Expected Distribution for Station -3 Ring 3",36,0.5,36.5);

  dbe->setCurrentFolder(folderPath+"BarrelPerLayer/");
  ExpLayerWm2= dbe->book1D("ExpLayerWm2","Expected Wheel - 2",6,0.5,6.5);
  ExpLayerWm1= dbe->book1D("ExpLayerWm1","Expected Wheel - 1",6,0.5,6.5);
  ExpLayerW0= dbe->book1D("ExpLayerW0","Expected Wheel 0",6,0.5,6.5);
  ExpLayerW1= dbe->book1D("ExpLayerW1","Expected Wheel 1",6,0.5,6.5);
  ExpLayerW2= dbe->book1D("ExpLayerW2","Expected Wheel 2",6,0.5,6.5);

  ObsLayerWm2= dbe->book1D("ObsLayerWm2","Observed Wheel - 2",6,0.5,6.5);
  ObsLayerWm1= dbe->book1D("ObsLayerWm1","Observed Wheel - 1",6,0.5,6.5);
  ObsLayerW0= dbe->book1D("ObsLayerW0","Observed Wheel 0",6,0.5,6.5);
  ObsLayerW1= dbe->book1D("ObsLayerW1","Observed Wheel 1",6,0.5,6.5);
  ObsLayerW2= dbe->book1D("ObsLayerW2","Observed Wheel 2",6,0.5,6.5);
}

void RPCEfficiencySecond::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){ }

void RPCEfficiencySecond::endRun(const edm::Run& r, const edm::EventSetup& iSetup){
  
  if(debug) std::cout <<"\t Getting the RPC Geometry"<<std::endl;

  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
  
  
  //Setting Labels in Summary Label.
  std::stringstream binLabel;
 
  if(debug) std::cout<<"Default -1 for Barrel GUI"<<std::endl;
  
  for(int x = 1;x<=12;x++){
    for(int y = 1;y<=21;y++){
      Wheelm2Summary->setBinContent(x,y,-1);
      Wheelm1Summary->setBinContent(x,y,-1);
      Wheel0Summary->setBinContent(x,y,-1);
      Wheel1Summary->setBinContent(x,y,-1);
      Wheel2Summary->setBinContent(x,y,-1);
    }
  }
 
  if(debug) std::cout<<"Default -1 for EndCap GUI"<<std::endl;
 
  for(int x = 1;x<=36;x++){
    for(int y = 1;y<=6;y++){
      Diskm3Summary->setBinContent(x,y,-1);
      Diskm2Summary->setBinContent(x,y,-1);
      Diskm1Summary->setBinContent(x,y,-1);
      Disk1Summary->setBinContent(x,y,-1);
      Disk2Summary->setBinContent(x,y,-1);
      Disk3Summary->setBinContent(x,y,-1);
    }
  }
 
  binLabel.str("");
 
  int indexWheel[5];
  for(int j=0;j<5;j++){
    indexWheel[j]=0;
  }
   
  int indexWheelf[5];
  for(int j=0;j<5;j++){
    indexWheelf[j]=0;
  }
 
  int indexDisk[6];
  for(int j=0;j<6;j++){
    indexDisk[j]=0;
  }
   
  int indexDiskf[6];
  for(int j=0;j<6;j++){
    indexDiskf[j]=0;
  }


  for(TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	RPCGeomServ rpcsrv(rpcId);
//	int sector = rpcId.sector();	
	std::string nameRoll = rpcsrv.name();
 

	if(debug){
	  if(meCollection.find(rpcId.rawId())==meCollection.end()){
	    std::cout<<"WARNING!!! Empty RecHit collection map"<<std::endl;
	  }
 	  std::cout<<rpcId<<std::endl;
	  //printing indexes
	  std::cout<<"indexWheel=";
	  for(int j=0;j<5;j++){
	    std::cout<<indexWheel[j]<<" ";
	  }
	  std::cout<<std::endl;
	  std::cout<<"indexWheelf=";
	  for(int j=0;j<5;j++){
	    std::cout<<indexWheelf[j]<<" ";
	  }
	  std::cout<<std::endl;
	  std::cout<<"indexDisk=";
	  for(int j=0;j<6;j++){
	    std::cout<<indexDisk[j]<<" ";
	  }
	  std::cout<<std::endl;
	  std::cout<<"indexDiskf=";
	  for(int j=0;j<6;j++){
	    std::cout<<indexDiskf[j]<<" ";
	  }
	  std::cout<<std::endl;
	}
  	
	if(rpcId.region()==0){
	  std::stringstream meIdRPC,  meIdDT; //,  bxDistroId;


	  meIdRPC<<folderPath<<"MuonSegEff/RPCDataOccupancyFromDT_"<<rpcId.rawId();
	  meIdDT<<folderPath<<"MuonSegEff/ExpectedOccupancyFromDT_"<<rpcId.rawId();

	  histoRPC = dbe->get(meIdRPC.str());
	  histoDT = dbe->get(meIdDT.str());
	  //  histoPRO = dbe->get(meIdPRO);
	 
	  int NumberWithOutPrediction=0;
	  double p = 0.;
	  double o = 0.;
// 	  float mybxhisto = 0.;
// 	  float mybxerror = 0.;
	  float ef = 0.;
	  float er = 0.;
	  float buffef = 0.;
	  float buffer = 0.;
	  float sumbuffef = 0.;
	  float sumbuffer = 0.;
	  float averageeff = 0.;
	  //float averageerr = 0.;
	  int NumberStripsPointed = 0;
	 
	  
	  if(histoRPC && histoDT){ // && BXDistribution){
	    if(debug) std::cout <<rpcsrv.name()<<std::endl;
	    
	    for(int i=1;i<=int((*r)->nstrips());++i){
	  
	      if(histoDT->getBinContent(i)!=0){
		if(debug) std::cout<<"Inside the If"<<std::endl;
		buffef = float(histoRPC->getBinContent(i))/float(histoDT->getBinContent(i));
		//	meMap[meIdPRO]->setBinContent(i,buffef); 
		buffer = sqrt(buffef*(1.-buffef)/float(histoDT->getBinContent(i)));
		//	meMap[meIdPRO]->setBinError(i,buffer);
		sumbuffef=sumbuffef+buffef;
		sumbuffer = sumbuffer + buffer*buffer;
		NumberStripsPointed++;
	      }else{
		NumberWithOutPrediction++;
	      }
	      if(debug) std::cout<<"\t Strip="<<i<<" RPC="<<histoRPC->getBinContent(i)<<" DT="<<histoDT->getBinContent(i)<<" buffef="<<buffef<<" buffer="<<buffer<<" sumbuffef="<<sumbuffef<<" sumbuffer="<<sumbuffer<<" NumberStripsPointed="<<NumberStripsPointed<<" NumberWithOutPrediction"<<NumberWithOutPrediction<<std::endl;
	    }
	    
	    p=histoDT->getTH1F()->Integral();
	    o=histoRPC->getTH1F()->Integral();
	    
	    if(NumberStripsPointed!=0){
	      averageeff = (sumbuffef/float(NumberStripsPointed))*100.;
	      //averageerr = sqrt(sumbuffer/float(NumberStripsPointed))*100.;
	    }
	    
	  } //////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  
	  int Ring = rpcId.ring();
	  
	  if(p!=0){
	    ef = float(o)/float(p); 
	    er = sqrt(ef*(1.-ef)/float(p));
	  }
	    
	  ef=ef*100;
	  er=er*100;


	   //Filling azimutal Wheel Histograms

	  int wheel = rpcId.ring();
	  int sector = rpcId.sector();
	  int region = rpcId.region();
	  
	  
	  if(region ==0){

	    int layer = 0;
	  	    
	    if(rpcId.station()==1&&rpcId.layer()==1) layer = 1;
	    else if(rpcId.station()==1&&rpcId.layer()==2) layer = 2;
	    else if(rpcId.station()==2&&rpcId.layer()==1) layer = 3;
	    else if(rpcId.station()==2&&rpcId.layer()==2)  layer = 4;
	    else if(rpcId.station()==3) layer = 5;
	    else if(rpcId.station()==4) layer = 6;

	    if(wheel==-2){ExsectorEffWm2->Fill(sector,p); OcsectorEffWm2->Fill(sector,o); ExpLayerWm2->Fill(layer, p); ObsLayerWm2->Fill(layer, o);}
	    else if(wheel==-1){ExsectorEffWm1->Fill(sector,p); OcsectorEffWm1->Fill(sector,o); ExpLayerWm1->Fill(layer, p); ObsLayerWm1->Fill(layer, o);}
	    else if(wheel==0){ExsectorEffW0->Fill(sector,p); OcsectorEffW0->Fill(sector,o); ExpLayerW0->Fill(layer, p); ObsLayerW0->Fill(layer, o);}
            else if(wheel==1){ExsectorEffW1->Fill(sector,p); OcsectorEffW1->Fill(sector,o); ExpLayerW1->Fill(layer, p); ObsLayerW1->Fill(layer, o);}
            else if(wheel==2){ExsectorEffW2->Fill(sector,p); OcsectorEffW2->Fill(sector,o); ExpLayerW2->Fill(layer, p); ObsLayerW2->Fill(layer, o);}
	  }

	    
	  std::string camera = rpcsrv.name();
	    
	  //float nopredictionsratio = (float(NumberWithOutPrediction)/float((*r)->nstrips()))*100.;

	  //Efficiency for Pigis Histos

	  if(debug) std::cout<<"Pigi "<<camera<<" "<<rpcsrv.shortname()<<" "
			     <<(*r)->id()<<std::endl;
	  
	  if(p > 100){//We need at least 100 predictions to fill the summary plot

			int xBin,yBin;
		  	xBin= (*r)->id().sector();
	  		rpcdqm::utils rollNumber;
	  		yBin = rollNumber.detId2RollNr((*r)->id());

	      		if((*r)->id().ring()==2) Wheel2Summary->setBinContent(xBin,yBin,averageeff);
	      		else if((*r)->id().ring()==1) Wheel1Summary->setBinContent(xBin,yBin,averageeff);
	      		else if((*r)->id().ring()==0) Wheel0Summary->setBinContent(xBin,yBin,averageeff);
	      		else if((*r)->id().ring()==-1) Wheelm1Summary->setBinContent(xBin,yBin,averageeff);
	      		else if((*r)->id().ring()==-2) Wheelm2Summary->setBinContent(xBin,yBin,averageeff);

	  }
 	  
	  //Near Side

	  //float maskedratio =0;

//	  if((sector==1||sector==2||sector==3||sector==10||sector==11||sector==12)){
	    if(Ring==-2){
	      EffDistroWm2->Fill(averageeff);
	      indexWheel[0]++;  
	      EffGlobWm2->setBinContent(indexWheel[0],ef);  
	      EffGlobWm2->setBinError(indexWheel[0],er);  
	      EffGlobWm2->setBinLabel(indexWheel[0],camera,1);

	    }else if(Ring==-1){
	      EffDistroWm1->Fill(averageeff);
	      indexWheel[1]++;  
	      EffGlobWm1->setBinContent(indexWheel[1],ef);  
	      EffGlobWm1->setBinError(indexWheel[1],er);  
	      EffGlobWm1->setBinLabel(indexWheel[1],camera,1);  

	    }else if(Ring==0){
	      EffDistroW0->Fill(averageeff);
	      indexWheel[2]++;  
	      EffGlobW0->setBinContent(indexWheel[2],ef);  
	      EffGlobW0->setBinError(indexWheel[2],er);  
	      EffGlobW0->setBinLabel(indexWheel[2],camera,1);  
	      
	    }else if(Ring==1){
	      EffDistroW1->Fill(averageeff);
	      indexWheel[3]++;  
	      EffGlobW1->setBinContent(indexWheel[3],ef);  
	      EffGlobW1->setBinError(indexWheel[3],er);  
	      EffGlobW1->setBinLabel(indexWheel[3],camera,1);  
	      
	    }else if(Ring==2){
	      EffDistroW2->Fill(averageeff);
	      indexWheel[4]++;
	      EffGlobW2->setBinContent(indexWheel[4],ef);
	      EffGlobW2->setBinError(indexWheel[4],er);
	      EffGlobW2->setBinLabel(indexWheel[4],camera,1);
	    }

	}else{//EndCap

	  std::stringstream meIdRPC,meIdCSC; //, bxDistroId;
	  std::string      meIdPRO;

	  
	  meIdRPC<<folderPath<<"MuonSegEff/RPCDataOccupancyFromCSC_"<<rpcId.rawId();
	  meIdCSC<<folderPath<<"MuonSegEff/ExpectedOccupancyFromCSC_"<<rpcId.rawId();

	  meIdPRO = "Profile_"+ rpcsrv.name();

	  histoRPC= dbe->get(meIdRPC.str());
	  histoCSC= dbe->get(meIdCSC.str());
	  //BXDistribution = dbe->get(bxDistroId.str());
	  		  
	  int NumberWithOutPrediction=0;
	  double p = 0;
	  double o = 0;
	  //	  float mybxhisto = 0;
	  //float mybxerror = 0;
	  float ef =0;
	  float er =0;
	  float buffef = 0;
	  float buffer = 0;
	  float sumbuffef = 0;
	  float sumbuffer = 0;
	  float averageeff = 0;
	  //float averageerr = 0;
	  int NumberStripsPointed = 0;


	  if(histoRPC && histoCSC) {// && BXDistribution){
	    if(debug) std::cout <<rpcsrv.name()<<std::endl;
	    
	    for(int i=1;i<=int((*r)->nstrips());++i){
	      if(histoCSC->getBinContent(i)!=0){
		if(debug) std::cout<<"Inside the If"<<std::endl;
		buffef = float(histoRPC->getBinContent(i))/float(histoCSC->getBinContent(i));
		//	meMap[meIdPRO]->setBinContent(i,buffef); 
		buffer = sqrt(buffef*(1.-buffef)/float(histoCSC->getBinContent(i)));
		//	meMap[meIdPRO]->setBinError(i,buffer);
		sumbuffef=sumbuffef+buffef;
		sumbuffer = sumbuffer + buffer*buffer;
		NumberStripsPointed++;
	      }else{
		NumberWithOutPrediction++;
	      }
	      
	      if(debug) std::cout<<"\t Strip="<<i<<" RPC="<<histoRPC->getBinContent(i)<<" CSC="<<histoCSC->getBinContent(i)<<" buffef="<<buffef<<" buffer="<<buffer<<" sumbuffef="<<sumbuffef<<" sumbuffer="<<sumbuffer<<" NumberStripsPointed="<<NumberStripsPointed<<" NumberWithOutPrediction"<<NumberWithOutPrediction<<std::endl;
	    }
	    p=histoCSC->getTH1F()->Integral();
	    o=histoRPC->getTH1F()->Integral();
	    
	    if(NumberStripsPointed!=0){
	      averageeff = (sumbuffef/float(NumberStripsPointed))*100.;
	      //averageerr = sqrt(sumbuffer/float(NumberStripsPointed))*100.;
	    }
	    
	   //  mybxhisto = 50.+BXDistribution->getMean()*10;
// 	    mybxerror = BXDistribution->getRMS()*10;
	  }
	  
	  int Disk = rpcId.station()*rpcId.region();

	  if(p!=0){
	    ef = float(o)/float(p); 
	    er = sqrt(ef*(1.-ef)/float(p));
	  }
	    
	  ef=ef*100;
	  er=er*100;

	   //Filling azimutal GregHistograms
	  
	  if(rpcId.region()==1){
	    if(rpcId.station()==1 && rpcId.ring()==2){ ExGregD1R2->Fill(rpcsrv.segment(),p);OcGregD1R2->Fill(rpcsrv.segment(),o);} 
	    else if(rpcId.station()==1 && rpcId.ring()==3){ ExGregD1R3->Fill(rpcsrv.segment(),p);OcGregD1R3->Fill(rpcsrv.segment(),o);}
	    else if(rpcId.station()==2 && rpcId.ring()==2){ ExGregD2R2->Fill(rpcsrv.segment(),p);OcGregD2R2->Fill(rpcsrv.segment(),o);}
	    else if(rpcId.station()==2 && rpcId.ring()==3){ ExGregD2R3->Fill(rpcsrv.segment(),p);OcGregD2R3->Fill(rpcsrv.segment(),o);}
	    else if(rpcId.station()==3 && rpcId.ring()==2){ ExGregD3R2->Fill(rpcsrv.segment(),p);OcGregD3R2->Fill(rpcsrv.segment(),o);}
	    else if(rpcId.station()==3 && rpcId.ring()==3){ ExGregD3R3->Fill(rpcsrv.segment(),p);OcGregD3R3->Fill(rpcsrv.segment(),o);}
	  }else if(rpcId.region()==-1){
	    if(rpcId.station()==1 && rpcId.ring()==2){ ExGregDm1R2->Fill(rpcsrv.segment(),p);OcGregDm1R2->Fill(rpcsrv.segment(),o);} 
	    else if(rpcId.station()==1 && rpcId.ring()==3){ ExGregDm1R3->Fill(rpcsrv.segment(),p);OcGregDm1R3->Fill(rpcsrv.segment(),o);}
	    else if(rpcId.station()==2 && rpcId.ring()==2){ ExGregDm2R2->Fill(rpcsrv.segment(),p);OcGregDm2R2->Fill(rpcsrv.segment(),o);}
	    else if(rpcId.station()==2 && rpcId.ring()==3){ ExGregDm2R3->Fill(rpcsrv.segment(),p);OcGregDm2R3->Fill(rpcsrv.segment(),o);}
	    else if(rpcId.station()==3 && rpcId.ring()==2){ ExGregDm3R2->Fill(rpcsrv.segment(),p);OcGregDm3R2->Fill(rpcsrv.segment(),o);}
	    else if(rpcId.station()==3 && rpcId.ring()==3){ ExGregDm3R3->Fill(rpcsrv.segment(),p);OcGregDm3R3->Fill(rpcsrv.segment(),o);}
	  }
	    
	  std::string camera = rpcsrv.name();
	    
	  //float nopredictionsratio = (float(NumberWithOutPrediction)/float((*r)->nstrips()))*100.;
	  

	  //Efficiency for Pigis Histos

	  if(debug) std::cout<<"Pigi "<<camera<<" "<<rpcsrv.shortname()<<" "
			     <<(*r)->id()<<std::endl;



	  if(p > 100){ //We need at least 100 predictions to fill the summary plot
	    RPCGeomServ RPCServ((*r)->id());
	    int xBin = RPCServ.segment();
	    int yBin= ((*r)->id().ring()-1)*3-(*r)->id().roll()+1;
	    if(Disk==-3) Diskm3Summary->setBinContent(xBin, yBin, averageeff);
	    else if(Disk==-2) Diskm2Summary->setBinContent(xBin, yBin, averageeff);
	    else if(Disk==-1) Diskm1Summary->setBinContent(xBin, yBin, averageeff);
	    else if(Disk==1) Disk1Summary->setBinContent(xBin, yBin, averageeff);
	    else if(Disk==2) Disk2Summary->setBinContent(xBin, yBin, averageeff);
	    else if(Disk==3) Disk3Summary->setBinContent(xBin, yBin, averageeff);
	  }

 	  //Near Side

	  //float maskedratio =0;

//	  if(sector==1||sector==2||sector==6){

	    if(Disk==-3){
	      EffDistroDm3->Fill(averageeff);
	      indexDisk[0]++;  
	      EffGlobDm3->setBinContent(indexDisk[0],ef);  
	      EffGlobDm3->setBinError(indexDisk[0],er);  
	      EffGlobDm3->setBinLabel(indexDisk[0],camera,1);


	    }else if(Disk==-2){
	      EffDistroDm2->Fill(averageeff);
	      indexDisk[1]++;  
	      EffGlobDm2->setBinContent(indexDisk[1],ef);  
	      EffGlobDm2->setBinError(indexDisk[1],er);  
	      EffGlobDm2->setBinLabel(indexDisk[1],camera,1);


	    }else if(Disk==-1){
	      EffDistroDm1->Fill(averageeff);
	      indexDisk[2]++;  
	      EffGlobDm1->setBinContent(indexDisk[2],ef);  
	      EffGlobDm1->setBinError(indexDisk[2],er);  
	      EffGlobDm1->setBinLabel(indexDisk[2],camera,1);  
	      

	    }else if(Disk==1){
	      EffDistroD1->Fill(averageeff);
	      indexDisk[3]++;  
	      EffGlobD1->setBinContent(indexDisk[3],ef);  
	      EffGlobD1->setBinError(indexDisk[3],er);  
	      EffGlobD1->setBinLabel(indexDisk[3],camera,1);  
	      

	    }else if(Disk==2){
	      EffDistroD2->Fill(averageeff);
	      indexDisk[4]++;
	      EffGlobD2->setBinContent(indexDisk[4],ef);
	      EffGlobD2->setBinError(indexDisk[4],er);
	      EffGlobD2->setBinLabel(indexDisk[4],camera,1);


	    }else if(Disk==3){
	      EffDistroD3->Fill(averageeff);
	      indexDisk[5]++;
	      EffGlobD3->setBinContent(indexDisk[5],ef);
	      EffGlobD3->setBinError(indexDisk[5],er);
	      EffGlobD3->setBinLabel(indexDisk[5],camera,1);

   
	    }

	}
      }
    }
  }

 float eff,N,err;
 int k;
 for(k=1;k<=36;k++){
   err=0; eff=0; N=ExGregD1R2->getBinContent(k);
   if(N!=0.){ eff = OcGregD1R2->getBinContent(k)/N; err=sqrt(eff*(1-eff)/N);}
   GregD1R2->setBinContent(k,eff); GregD1R2->setBinError(k,err);
   
   err=0; eff=0; N=ExGregD1R3->getBinContent(k);
   if(N!=0.){eff = OcGregD1R3->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregD1R3->setBinContent(k,eff); GregD1R3->setBinError(k,err);
   
   err=0; eff=0; N=ExGregD2R2->getBinContent(k);
   if(N!=0.){ eff = OcGregD2R2->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregD2R2->setBinContent(k,eff); GregD2R2->setBinError(k,err);
   
   err=0; eff=0; N=ExGregD2R3->getBinContent(k);
   if(N!=0.){ eff = OcGregD2R3->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregD2R3->setBinContent(k,eff); GregD2R3->setBinError(k,err);
   
   err=0; eff=0; N=ExGregD3R2->getBinContent(k);
   if(N!=0.){ eff = OcGregD3R2->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregD3R2->setBinContent(k,eff); GregD3R2->setBinError(k,err);
   
   err=0; eff=0; N=ExGregD3R3->getBinContent(k);
   if(N!=0.){ eff = OcGregD3R3->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregD3R3->setBinContent(k,eff); GregD3R3->setBinError(k,err);

   err=0; eff=0; N=ExGregDm1R2->getBinContent(k);
   if(N!=0.){ eff = OcGregDm1R2->getBinContent(k)/N; err=sqrt(eff*(1-eff)/N);}
   GregDm1R2->setBinContent(k,eff); GregDm1R2->setBinError(k,err);
   
   err=0; eff=0; N=ExGregDm1R3->getBinContent(k);
   if(N!=0.){eff = OcGregDm1R3->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregDm1R3->setBinContent(k,eff); GregDm1R3->setBinError(k,err);
   
   err=0; eff=0; N=ExGregDm2R2->getBinContent(k);
   if(N!=0.){ eff = OcGregDm2R2->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregDm2R2->setBinContent(k,eff); GregDm2R2->setBinError(k,err);
   
   err=0; eff=0; N=ExGregDm2R3->getBinContent(k);
   if(N!=0.){ eff = OcGregDm2R3->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregDm2R3->setBinContent(k,eff); GregDm2R3->setBinError(k,err);
   
   err=0; eff=0; N=ExGregDm3R2->getBinContent(k);
   if(N!=0.){ eff = OcGregDm3R2->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregDm3R2->setBinContent(k,eff); GregDm3R2->setBinError(k,err);
   
   err=0; eff=0; N=ExGregDm3R3->getBinContent(k);
   if(N!=0.){ eff = OcGregDm3R3->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
   GregDm3R3->setBinContent(k,eff); GregDm3R3->setBinError(k,err);
 }

  for(k=1;k<=12;k++){
    err=0; eff=0; N=ExsectorEffWm2->getBinContent(k);
    if(N!=0.){ eff = OcsectorEffWm2->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
    sectorEffWm2->setBinContent(k,eff); sectorEffWm2->setBinError(k,err);
    //std::cout<<N<<" "<<OcsectorEffWm2->getBinContent(k)<<" "<<eff<<" "<<err<<std::endl;

    err=0; eff=0; N=ExsectorEffWm1->getBinContent(k);
    if(N!=0.){ eff = OcsectorEffWm1->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
    sectorEffWm1->setBinContent(k,eff); sectorEffWm1->setBinError(k,err);
    //std::cout<<N<<" "<<OcsectorEffWm1->getBinContent(k)<<" "<<eff<<" "<<err<<std::endl;

    err=0; eff=0; N=ExsectorEffW0->getBinContent(k);
    if(N!=0.){ eff = OcsectorEffW0->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
    sectorEffW0->setBinContent(k,eff); sectorEffW0->setBinError(k,err);
    //std::cout<<N<<" "<<OcsectorEffW0->getBinContent(k)<<" "<<eff<<" "<<err<<std::endl;

    err=0; eff=0; N=ExsectorEffW1->getBinContent(k);
    if(N!=0.){ eff = OcsectorEffW1->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
    sectorEffW1->setBinContent(k,eff); sectorEffW1->setBinError(k,err);
    //std::cout<<N<<" "<<OcsectorEffW1->getBinContent(k)<<" "<<eff<<" "<<err<<std::endl;

    err=0; eff=0; N=ExsectorEffW2->getBinContent(k);
    if(N!=0.){ eff = OcsectorEffW2->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
    sectorEffW2->setBinContent(k,eff); sectorEffW2->setBinError(k,err);
    //std::cout<<N<<" "<<OcsectorEffW2->getBinContent(k)<<" "<<eff<<" "<<err<<std::endl;
  }

  //Ranges for Both
  //Barrel

  if(barrel){
    EffGlobWm2->setAxisRange(-4.,100.,2);
    EffGlobWm1->setAxisRange(-4.,100.,2);
    EffGlobW0->setAxisRange(-4.,100.,2);
    EffGlobW1->setAxisRange(-4.,100.,2);
    EffGlobW2->setAxisRange(-4.,100.,2);
  }

  //EndCap

  if(endcap){
    EffGlobDm3->setAxisRange(-4.,100.,2);
    EffGlobDm2->setAxisRange(-4.,100.,2);
    EffGlobDm1->setAxisRange(-4.,100.,2);
    EffGlobD1->setAxisRange(-4.,100.,2);
    EffGlobD2->setAxisRange(-4.,100.,2);
    EffGlobD3->setAxisRange(-4.,100.,2);


  }

  //Title for Both

  //Barrel
  if(barrel){
    EffGlobWm2->setAxisTitle("%",2);
    EffGlobWm1->setAxisTitle("%",2);
    EffGlobW0->setAxisTitle("%",2);
    EffGlobW1->setAxisTitle("%",2);
    EffGlobW2->setAxisTitle("%",2);

  }
  //EndCap

  if(endcap){
    EffGlobDm3->setAxisTitle("%",2);
    EffGlobDm2->setAxisTitle("%",2);
    EffGlobDm1->setAxisTitle("%",2);
    EffGlobD1->setAxisTitle("%",2);
    EffGlobD2->setAxisTitle("%",2);
    EffGlobD3->setAxisTitle("%",2);


  }
  

  if(SaveFile){
    std::cout<<"Saving RootFile"<<std::endl;
    dbe->save(NameFile);
  }
    
}

void RPCEfficiencySecond::endJob(){}

