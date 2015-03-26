/***************************************
Original Author:Camilo Carrillo
****************************************/
#include <sstream>

#include "DQM/RPCMonitorDigi/interface/utils.h"
#include "DQM/RPCMonitorClient/interface/RPCEfficiencySecond.h"
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

RPCEfficiencySecond::RPCEfficiencySecond(const edm::ParameterSet& iConfig){

  folderPath  = iConfig.getUntrackedParameter<std::string>("folderPath","RPC/RPCEfficiency/"); 
  numberOfDisks_ =   iConfig.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  innermostRings_ = iConfig.getUntrackedParameter<int>("NumberOfInnermostEndcapRings", 2);

  init_ = false;

}
 
RPCEfficiencySecond::~RPCEfficiencySecond(){}

void RPCEfficiencySecond::beginJob(){}

void RPCEfficiencySecond::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter,  edm::LuminosityBlock const & lb, edm::EventSetup const& iSetup){

  if(!init_){
    
    LogDebug("rpcefficiencysecond")<<"Getting the RPC Geometry";    
  
    iSetup.get<MuonGeometryRecord>().get(rpcGeo_);   
    init_= true;

  }
}


void  RPCEfficiencySecond::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
 

  rpcdqm::utils rpcUtils;
  //Barrel
  std::stringstream folderName;
  std::stringstream MeName; std::stringstream MeTitle;
  for (int w =-2; w<=2; w++){
    folderName.str("");
    folderName<<folderPath<<"Wheel_"<<w;
    ibooker.setCurrentFolder(folderName.str());
    MeName.str("");
    MeName<<"EffDistroWheel_"<<w;
    MeTitle.str("");
    MeTitle<<"Efficiency Distribution for Wheel "<<w;
    EffDistroW[w+2]=ibooker.book1D(MeName.str(),MeTitle.str(),20,0.5,100.5);
    MeName.str("");
    MeName<<"GlobEfficiencyWheel_"<<w;
    MeTitle.str("");
    MeTitle<<"Efficiency Wheel "<<w;
    EffGlobW[w+2]=ibooker.book1D(MeName.str(),MeTitle.str(),206,0.5,206.5);
    ibooker.setCurrentFolder(folderPath);
    MeName.str("");
    MeName<<"Efficiency_Roll_vs_Sector_Wheel_"<<w;
    WheelSummary[w+2] = ibooker.book2D(MeName.str(),MeName.str(),12, 0.5,12.5, 21, 0.5, 21.5);
    rpcUtils.labelXAxisSector( WheelSummary[w+2] );
    rpcUtils.labelYAxisRoll( WheelSummary[w+2], 0, w, true);
    ibooker.setCurrentFolder(folderPath+"Azimutal/");
    MeName.str("");
    MeName<<"AzimutalDistroW"<<w;
    MeTitle.str("");
    MeTitle<<"Efficiency per Sector Wheel "<<w;
    sectorEffW[w+2]= ibooker.book1D(MeName.str(),MeTitle.str(),12,0.5,12.5);
    MeName.str("");
    MeName<<"AzimutalDistroW"<<w<<"Ex";
    MeTitle.str("");
    MeTitle<<"Expected per Sector Wheel "<<w;
    ExsectorEffW[w+2]= ibooker.book1D(MeName.str(),MeTitle.str(),12,0.5,12.5);
    MeName.str("");
    MeName<<"AzimutalDistroW"<<w<<"Oc";
    MeTitle.str("");
    MeTitle<<"Occupancy per Sector Wheel "<<w;
    OcsectorEffW[w+2]= ibooker.book1D(MeName.str(),MeTitle.str(),12,0.5,12.5);
    ibooker.setCurrentFolder(folderPath+"BarrelPerLayer/");
    MeName.str("");
    MeName<<"ExpLayerW"<<w;
    MeTitle.str("");
    MeTitle<<"Expected Wheel "<<w;
    ExpLayerW[w+2]= ibooker.book1D(MeName.str(),MeTitle.str(),6,0.5,6.5);
    MeName.str("");
    MeName<<"ObsLayerW"<<w;
    MeTitle.str("");
    MeTitle<<"Observed Wheel "<<w;
    ObsLayerW[w+2]= ibooker.book1D(MeName.str(),MeTitle.str(),6,0.5,6.5);
  }
  //EndCap
  int index = 0;
  for (int d = (-1 *numberOfDisks_); d<=numberOfDisks_; d++){
    if (d==0) {continue;}
    folderName.str("");
    folderName<<folderPath<<"Disk_"<<d;
    ibooker.setCurrentFolder(folderName.str());
    MeName.str("");
    MeName<<"EffDistroDisk_"<<d;
    MeTitle.str("");
    MeTitle<<"Efficiency Distribution Disk "<<d;
    EffDistroD[index]=ibooker.book1D(MeName.str(),MeTitle.str(),20,0.5,100.5);
    MeName.str("");
    MeName<<"GlobEfficiencyDisk_"<<d;
    MeTitle.str("");
    MeTitle<<"Efficiency Disk "<<d;
    EffGlobD[index] = ibooker.book1D(MeName.str(),MeTitle.str(),218,0.5,218.5);
    ibooker.setCurrentFolder(folderPath);
    MeName.str("");
    MeName<<"Efficiency_Roll_vs_Segment_Disk_"<<d;
    DiskSummary[index] = ibooker.book2D(MeName.str(),MeName.str(),36,0.5,36.5,6,0.5,6.5);
    rpcUtils.labelXAxisSegment(DiskSummary[index]);
    rpcUtils.labelYAxisRing(DiskSummary[index], innermostRings_ , true);
    ibooker.setCurrentFolder(folderPath+"Azimutal/");
    MeName.str("");
    MeName<<"GregDistroR2D"<<d;
    MeTitle.str("");
    MeTitle<<"Efficiency for Station "<<d<<" Ring 2";
    GregR2D[index]= ibooker.book1D(MeName.str(),MeTitle.str(),36,0.5,36.5);
    MeName.str("");
    MeName<<"GregDistroR3D"<<d;
    MeTitle.str("");
    MeTitle<<"Efficiency for Station "<<d<<" Ring 3";
    GregR3D[index]= ibooker.book1D(MeName.str(),MeTitle.str(),36,0.5,36.5);
    MeName.str("");
    MeName<<"OcGregDistroR2D"<<d;
    MeTitle.str("");
    MeTitle<<"Occupancy Distribution for Station "<<d<<" Ring 2";
    OcGregR2D[index]= ibooker.book1D(MeName.str(),MeTitle.str(),36,0.5,36.5);
    MeName.str("");
    MeName<<"OcGregDistroR3D"<<d;
    MeTitle.str("");
    MeTitle<<"Occupancy Distribution for Station "<<d<<" Ring 3";
    OcGregR3D[index]= ibooker.book1D(MeName.str(),MeTitle.str(),36,0.5,36.5);
    MeName.str("");
    MeName<<"ExGregDistroR2D"<<d;
    MeTitle.str("");
    MeTitle<<"Expected Distribution for Station "<<d<<" Ring 2";
    ExGregR2D[index]= ibooker.book1D(MeName.str(),MeTitle.str(),36,0.5,36.5);
    MeName.str("");
    MeName<<"ExGregDistroR3D"<<d;
    MeTitle.str("");
    MeTitle<<"Expected Distribution for Station "<<d<<" Ring 3";
    ExGregR3D[index]= ibooker.book1D(MeName.str(),MeTitle.str(),36,0.5,36.5);
    index++;
  }
  
  
  
  LogDebug("rpcefficiencysecond")<<"Getting the RPC Geometry";
  
  
  //Setting Labels in Summary Label.
  std::stringstream binLabel;
  for (int w = -2; w<=2 ;w++){
    for(int x = 1;x<=12;x++){
      for(int y = 1;y<=21;y++){
	WheelSummary[w+2]->setBinContent(x,y,-1);
      }
    }
  }
  for (int d = 0 ; d<(numberOfDisks_*2); d++){
    for(int x = 1;x<=36;x++){
      for(int y = 1;y<=6;y++){
	DiskSummary[d]->setBinContent(x,y,-1);
      }
    }
  }
  binLabel.str("");
  int indexWheel[5];
  for(int j=0;j<5;j++){
    indexWheel[j]=0;
  }
  int indexDisk[10];
  for(int j=0;j<10;j++){
    indexDisk[j]=0;
  }
  for(TrackingGeometry::DetContainer::const_iterator it=rpcGeo_->dets().begin(); it!=rpcGeo_->dets().end();it++){
    if(dynamic_cast< const RPCChamber* >( *it ) != 0 ){
      const RPCChamber* ch = dynamic_cast< const RPCChamber* >( *it );
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	RPCGeomServ rpcsrv(rpcId);
	// std::string nameRoll = rpcsrv.name();
	std::string camera = rpcsrv.name();
	//Breaking down the geometry
	int region = rpcId.region();
	int wheel = rpcId.ring(); int ring = rpcId.ring();
	int sector = rpcId.sector();
	int station = rpcId.station();
	int geolayer = rpcId.layer();

	RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure(); 
	std::string folder = folderPath + "MuonSegEff/" + folderStr->folderStructure(rpcId);


	if(region==0){//Barrel
	  std::stringstream meIdRPC, meIdDT; //, bxDistroId;


	  meIdRPC<<folder<<"/RPCDataOccupancyFromDT_"<<rpcId.rawId();
	  meIdDT<<folder<<"/ExpectedOccupancyFromDT_"<<rpcId.rawId();
	  histoRPC = igetter.get(meIdRPC.str());
	  histoDT = igetter.get(meIdDT.str());
	  int NumberWithOutPrediction=0;
	  double p = 0.;
	  double o = 0.;
	  double ef = 0.;
	  double er = 0.;
	  double buffef = 0.;
	  double buffer = 0.;
	  double sumbuffef = 0.;
	  double sumbuffer = 0.;
	  double averageeff = 0.;
	  int NumberStripsPointed = 0;
	  if(histoRPC && histoDT){
	    for(int i=1;i<=int((*r)->nstrips());++i){
	      if(histoDT->getBinContent(i)!=0){
		LogDebug("rpcefficiencysecond")<<"Inside the If";
		buffef = double(histoRPC->getBinContent(i))/double(histoDT->getBinContent(i));
		buffer = sqrt(buffef*(1.-buffef)/double(histoDT->getBinContent(i)));
		sumbuffef=sumbuffef+buffef;
		sumbuffer = sumbuffer + buffer*buffer;
		NumberStripsPointed++;
	      }else{
		NumberWithOutPrediction++;
	      }
	      LogDebug("rpcefficiencysecond")<<"Strip="<<i<<" RPC="<<histoRPC->getBinContent(i)<<" DT="<<histoDT->getBinContent(i)<<" buffef="<<buffef<<" sumbuffef="<<sumbuffef<<" NumberStripsPointed="<<NumberStripsPointed<<" NumberWithOutPrediction"<<NumberWithOutPrediction;
	    }
	    p=histoDT->getTH1F()->Integral();
	    o=histoRPC->getTH1F()->Integral();
	    if(NumberStripsPointed!=0){
	      averageeff = (sumbuffef/double(NumberStripsPointed))*100.;
	      //averageerr = sqrt(sumbuffer/double(NumberStripsPointed))*100.;
	    }
	  } //////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  // int Ring = rpcId.ring();
	  if(p!=0){
	    ef = double(o)/double(p);
	    er = sqrt(ef*(1.-ef)/double(p));
	  }
	  ef=ef*100;
	  er=er*100;
	  //Filling azimutal Wheel Histograms
	  int layer = 0;
	  if(station==1&&geolayer==1) layer = 1;
	  else if(station==1&&geolayer==2) layer = 2;
	  else if(station==2&&geolayer==1) layer = 3;
	  else if(station==2&&geolayer==2) layer = 4;
	  else if(station==3) layer = 5;
	  else if(station==4) layer = 6;
	  ExsectorEffW[wheel+2]->Fill(sector,p); OcsectorEffW[wheel+2]->Fill(sector,o);
	  ExpLayerW[wheel+2]->Fill(layer, p); ObsLayerW[wheel+2]->Fill(layer, o);
	  LogDebug("rpcefficiencysecond")<<"Pigi "<<camera<<" "<<rpcsrv.shortname()<<" "<<(*r)->id();
	  if(p > 100){//We need at least 100 predictions to fill the summary plot
	    int xBin,yBin;
	    xBin= (*r)->id().sector();
	    rpcdqm::utils rollNumber;
	    yBin = rollNumber.detId2RollNr((*r)->id());
	    WheelSummary[wheel+2]->setBinContent(xBin,yBin,averageeff);
	  }
	  EffDistroW[wheel+2]->Fill(averageeff);
	  indexWheel[wheel+2]++;
	  EffGlobW[wheel+2]->setBinContent(indexWheel[wheel+2],ef);
	  EffGlobW[wheel+2]->setBinError(indexWheel[wheel+2],er);
	  EffGlobW[wheel+2]->setBinLabel(indexWheel[wheel+2],camera,1);
	}else{//EndCap
	  std::stringstream meIdRPC,meIdCSC; //, bxDistroId;
	  std::string meIdPRO;
	  meIdRPC<<folder<<"/RPCDataOccupancyFromCSC_"<<rpcId.rawId();
	  meIdCSC<<folder<<"/ExpectedOccupancyFromCSC_"<<rpcId.rawId();
	  histoRPC= igetter.get(meIdRPC.str());
	  histoCSC= igetter.get(meIdCSC.str());
	  
	  int NumberWithOutPrediction=0;
	  double p = 0;
	  double o = 0;
	  double ef =0;
	  double er =0;
	  double buffef = 0;
	  double buffer = 0;
	  double sumbuffef = 0;
	  double sumbuffer = 0;
	  double averageeff = 0;
	  int NumberStripsPointed = 0;
	  if(histoRPC && histoCSC) {
	    LogDebug("rpcefficiencysecond")<<rpcsrv.name();
	    for(int i=1;i<=int((*r)->nstrips());++i){
	      if(histoCSC->getBinContent(i)!=0){
		LogDebug("rpcefficiencysecond")<<"Inside the If";
		buffef = double(histoRPC->getBinContent(i))/double(histoCSC->getBinContent(i));
		buffer = sqrt(buffef*(1.-buffef)/double(histoCSC->getBinContent(i)));
		sumbuffef=sumbuffef+buffef;
		sumbuffer = sumbuffer + buffer*buffer;
		NumberStripsPointed++;
	      }else{
		NumberWithOutPrediction++;
	      }
	      LogDebug("rpcefficiencysecond")<<"Strip="<<i<<" RPC="<<histoRPC->getBinContent(i)<<" CSC="<<histoCSC->getBinContent(i)<<" buffef="<<buffef<<" buffer="<<buffer<<" sumbuffef="<<sumbuffef<<" sumbuffer="<<sumbuffer<<" NumberStripsPointed="<<NumberStripsPointed<<" NumberWithOutPrediction"<<NumberWithOutPrediction;
	    }
	    p=histoCSC->getTH1F()->Integral();
	    o=histoRPC->getTH1F()->Integral();
	    if(NumberStripsPointed!=0){
	      averageeff = (sumbuffef/double(NumberStripsPointed))*100.;
	    }
	  }
	  int Disk = station*region;
	  if(Disk > numberOfDisks_ || Disk<-numberOfDisks_){continue;} //remove strange disk numbers!!
	  int dIndex = -1;
	  if(region == -1){
	    dIndex = Disk+numberOfDisks_;
	  }else if(region == 1){
	    dIndex = Disk+numberOfDisks_-1;
	  }
	  if( dIndex<0 || dIndex>= numberOfDisks_*2){continue;} //extra check on disk numering
	  if(p!=0){
	    ef = double(o)/double(p);
	    er = sqrt(ef*(1.-ef)/double(p));
	  }
	  ef=ef*100;
	  er=er*100;
	  if(ring==2){
	    if (ExGregR2D[dIndex] && OcGregR2D[dIndex] ){
	      ExGregR2D[dIndex]->Fill(rpcsrv.segment(),p);
	      OcGregR2D[dIndex]->Fill(rpcsrv.segment(),o);
	    }
	  } else if(ring==3){
	    ExGregR3D[dIndex]->Fill(rpcsrv.segment(),p);
	    OcGregR3D[dIndex]->Fill(rpcsrv.segment(),o);
	  }
	  if(p > 100){ //We need at least 100 predictions to fill the summary plot
	    RPCGeomServ RPCServ((*r)->id());
	    int xBin = RPCServ.segment();
	    int yBin= (ring-1)*3-(*r)->id().roll()+1;
	    DiskSummary[dIndex]->setBinContent(xBin, yBin, averageeff);
	  }
	  if ( EffDistroD[dIndex]){EffDistroD[dIndex]->Fill(averageeff);}
	  indexDisk[dIndex]++;
	  if ( EffGlobD[dIndex]){
	    EffGlobD[dIndex]->setBinContent(indexDisk[dIndex],ef);
	    EffGlobD[dIndex]->setBinError(indexDisk[dIndex],er);
	    EffGlobD[dIndex]->setBinLabel(indexDisk[dIndex],camera,1);
	  }
	}
	delete folderStr;
      }
    }
  

  }
  double eff,N,err;
  int k;
  for (int d = 0; d<(numberOfDisks_*2); d++){
    for(k=1;k<=36;k++){
      err=0; eff=0; N=ExGregR2D[d]->getBinContent(k);
      if(N!=0.){ eff = OcGregR2D[d]->getBinContent(k)/N; err=sqrt(eff*(1-eff)/N);}
      GregR2D[d]->setBinContent(k,eff); GregR2D[d]->setBinError(k,err);
      err=0; eff=0; N=ExGregR3D[d]->getBinContent(k);
      if(N!=0.){ eff = OcGregR3D[d]->getBinContent(k)/N; err=sqrt(eff*(1-eff)/N);}
      GregR3D[d]->setBinContent(k,eff); GregR3D[d]->setBinError(k,err);
    }
  }
  for (int w =-2; w<=2; w++){
    for(k=1;k<=12;k++){
      err=0; eff=0; N=ExsectorEffW[w+2]->getBinContent(k);
      if(N!=0.){ eff = OcsectorEffW[w+2]->getBinContent(k)/N;err=sqrt(eff*(1-eff)/N);}
      sectorEffW[w+2]->setBinContent(k,eff); sectorEffW[w+2]->setBinError(k,err);
    }
  }
  //Ranges for Both
  //Barrel
  for (int w=-2; w<=2; w++){
    EffGlobW[w+2]->setAxisRange(-4.,100.,2);
    EffGlobW[w+2]->setAxisTitle("%",2);
  }
  for (int d=0; d<(numberOfDisks_*2); d++){
    EffGlobD[d]->setAxisRange(-4.,100.,2);
    EffGlobD[d]->setAxisTitle("%",2);
  }


}

