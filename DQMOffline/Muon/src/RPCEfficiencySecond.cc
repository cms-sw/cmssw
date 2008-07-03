/***************************************
Author: 
Camilo Carrillo
Universidad de los Andes Bogota Colombia
camilo.carrilloATcern.ch
****************************************/

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "FWCore/Framework/interface/ESHandle.h"

#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include<string>
#include<fstream>
#include "DQMOffline/Muon/interface/RPCEfficiencySecond.h"

//Root
#include "TFile.h"
#include "TF1.h"
#include "TH1F.h"
#include "TH1.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TMath.h"


//
// class decleration
//

class TFile;


RPCEfficiencySecond::RPCEfficiencySecond(const edm::ParameterSet& iConfig){
   //now do what ever initialization is needed
  dbe = edm::Service<DQMStore>().operator->();

}


RPCEfficiencySecond::~RPCEfficiencySecond()
{}


void 
RPCEfficiencySecond::beginJob(const edm::EventSetup&)
{

}


void
RPCEfficiencySecond::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  std::cout <<"\t Getting the RPC Geometry"<<std::endl;
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);


  dbe->setCurrentFolder("Muons/RPCEfficiency");

  EffGlobWm2=dbe->book1D("GlobEfficiencyWheel_-2near","Efficiency Near Wheel -2 ",105,0.5,105.5);
  EffGlobWm1= dbe->book1D("GlobEfficiencyWheel_-1near","Efficiency Near Wheel -1",105,0.5,105.5);
  EffGlobW0 = dbe->book1D("GlobEfficiencyWheel_0near","Efficiency Near Wheel 0",105,0.5,105.5);
  EffGlobW1 = dbe->book1D("GlobEfficiencyWheel_1near","Efficiency Near Wheel 1",105,0.5,105.5);
  EffGlobW2 = dbe->book1D("GlobEfficiencyWheel_2near","Efficiency Near Wheel 2",105,0.5,105.5);

  EffGlobWm2far=dbe->book1D("GlobEfficiencyWheel_-2far","Efficiency Far Wheel -2",101,0.5,101.5);
  EffGlobWm1far=dbe->book1D("GlobEfficiencyWheel_-1far","Efficiency Far Wheel -1",101,0.5,101.5);
  EffGlobW0far =dbe->book1D("GlobEfficiencyWheel_0far","Efficiency Far Wheel 0",101,0.5,101.5);
  EffGlobW1far =dbe->book1D("GlobEfficiencyWheel_1far","Efficiency Far Wheel 1",101,0.5,101.5);
  EffGlobW2far =dbe->book1D("GlobEfficiencyWheel_2far","Efficiency Far Wheel 2",101,0.5,101.5);

  BXGlobWm2= dbe->book1D("GlobBXWheel_-2near","BX Near Wheel -2",105,0.5,105.5);
  BXGlobWm1= dbe->book1D("GlobBXWheel_-1near","BX Near Wheel -1",105,0.5,105.5);
  BXGlobW0 = dbe->book1D("GlobBXWheel_0near","BX Near Wheel 0",105,0.5,105.5);
  BXGlobW1 = dbe->book1D("GlobBXWheel_1near","BX Near Wheel 1",105,0.5,105.5);
  BXGlobW2 = dbe->book1D("GlobBXWheel_2near","BX Near Wheel 2",105,0.5,105.5);
  
  BXGlobWm2far= dbe->book1D("GlobBXWheel_-2far","BX Far Wheel -2",101,0.5,101.5);
  BXGlobWm1far= dbe->book1D("GlobBXWheel_-1far","BX Far Wheel -1",101,0.5,101.5);
  BXGlobW0far = dbe->book1D("GlobBXWheel_0far","BX Far Wheel 0",101,0.5,101.5);
  BXGlobW1far = dbe->book1D("GlobBXWheel_1far","BX Far Wheel 1",101,0.5,101.5);
  BXGlobW2far = dbe->book1D("GlobBXWheel_2far","BX Far Wheel 2",101,0.5,101.5);

  MaskedGlobWm2= dbe->book1D("GlobMaskedWheel_-2near","Masked Near Wheel -2",105,0.5,105.5);
  MaskedGlobWm1= dbe->book1D("GlobMaskedWheel_-1near","Masked Near Wheel -1",105,0.5,105.5);
  MaskedGlobW0 = dbe->book1D("GlobMaskedWheel_0near","Masked Near Wheel 0",105,0.5,105.5);
  MaskedGlobW1 = dbe->book1D("GlobMaskedWheel_1near","Masked Near Wheel 1",105,0.5,105.5);
  MaskedGlobW2 = dbe->book1D("GlobMaskedWheel_2near","Masked Near Wheel 2",105,0.5,105.5);

  MaskedGlobWm2far= dbe->book1D("GlobMaskedWheel_-2far","Masked Far Wheel -2",101,0.5,101.5);
  MaskedGlobWm1far= dbe->book1D("GlobMaskedWheel_-1far","Masked Far Wheel -1",101,0.5,101.5);
  MaskedGlobW0far = dbe->book1D("GlobMaskedWheel_0far","Masked Far Wheel 0",101,0.5,101.5);
  MaskedGlobW1far = dbe->book1D("GlobMaskedWheel_1far","Masked Far Wheel 1",101,0.5,101.5);
  MaskedGlobW2far = dbe->book1D("GlobMaskedWheel_2far","Masked Far Wheel 2",101,0.5,101.5);
  
  int indexWheel[5];
  for(int j=0;j<5;j++){
    indexWheel[j]=0;
  }
  
  int indexWheelf[5];
  for(int j=0;j<5;j++){
    indexWheelf[j]=0;
  }

  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	RPCGeomServ rpcsrv(rpcId);
		
	int NumberMasked=0;
	
	int sector = rpcId.sector();
  	
	if(rpcId.region()==0){
	  
	  char detUnitLabel[128];

	  char meIdRPC [128];
	  char meIdDT [128];
	  
	  char bxDistroId [128];

	  char meIdRealRPC[128];
	  
	  std::string regionName;
	  std::string ringType;
	  char  folder[120];
	  
	  
	  if(rpcId.region()==0){
	    regionName="Barrel";
	    ringType="Wheel";
	  }
	  else{
	    ringType="Disk";
	    if(rpcId.region() == -1) regionName="Endcap-";
	    if(rpcId.region() ==  1) regionName="Endcap+";
	  }
	  
	  sprintf(folder,"RPC/MuonSegEff/%s/%s_%d/station_%d/sector_%d",
		  regionName.c_str(),ringType.c_str(),rpcId.ring(),rpcId.station(),rpcId.sector());
	  sprintf(detUnitLabel ,"%s",rpcsrv.name().c_str());
	
	  sprintf(meIdRPC,"%s/RPCDataOccupancyFromDT_%s",folder,detUnitLabel);
	
	  sprintf(meIdDT,"%s/ExpectedOccupancyFromDT_%s",folder,detUnitLabel);
	  
	  sprintf(bxDistroId,"%s/BXDistribution_%s",folder,detUnitLabel);
	  sprintf(meIdRealRPC,"%s/RealDetectedOccupancyFromDT_%s",folder,detUnitLabel);  
	  
	  std::cout<<meIdRealRPC<<std::endl;

	  histoRPC = dbe->get("RPC/MuonSegEff/Barrel/Wheel_0/station_2/sector_11/RealDetectedOccupancyFromDT_W+0_RB2in_S11_Backward");
	  
	  if(!histoRPC)std::cout<<"NADA!!!!"<<std::endl;

	  histoRPC= dbe->get(meIdRPC);
	  histoDT= dbe->get(meIdDT);
	  BXDistribution = dbe->get(bxDistroId);
	  histoRealRPC = dbe->get(meIdRealRPC);
	  	  
	  std::cout<<"Before If..."<<std::endl;

	  if(!histoRPC)std::cout<<"false for histoRPC"<<std::endl;
	  if(!histoDT)std::cout<<"false for histoDT"<<std::endl;
	  if(!BXDistribution)std::cout<<"false for BXDistribution"<<std::endl;
	  if(!histoRealRPC)std::cout<<"false for histoRealRPC"<<std::endl;

	  if(histoRPC && histoDT && BXDistribution && histoRealRPC)
	    {
	    
	    std::cout<<"No empty Histogram"<<std::endl;
	    
	    bool somenthing1D = false;
	    
  	    for(unsigned int i=1;i<=int((*r)->nstrips());++i){
	      if(histoDT->getBinContent(i) != 0){
		//float eff = histoRPC->getBinContent(i)/histoDT->getBinContent(i);
		//float erreff = sqrt(eff*(1-eff)/histoDT->getBinContent(i));
		somenthing1D = true;
		std::cout<<"Bin Content"<<histoDT->getBinContent(i)<<std::endl;
	      }
	      if(histoRealRPC->getBinContent(i)==0) NumberMasked++;
	    }
	    
	    if(somenthing1D){
	      //histoRPC->Write();
	      //histoDT->Write();
	      //sprintf(namefile1D,"results/Efficiency/profile.%s.png",detUnitLabel);
	      //Ca1->SaveAs(namefile1D);
	      //Ca1->Clear();
	    }
	    
	    if(BXDistribution->getTH1F()->Integral()!=0){
	      //sprintf(bxFileName,"results/mydqm/bxDistribution.%s.png",detUnitLabel);
	      //Ca1->SaveAs(bxFileName);
	      //Ca1->Clear();
	    }
	    
	    
	    //Global Efficiency per Wheel
	    
	    int Ring = rpcId.ring();
	    
	    float ef =0;
	    float er =0;
	    
	    double p=histoDT->getTH1F()->Integral();
	    double o=histoRPC->getTH1F()->Integral();
	    
	    if(p!=0){
	      ef = float(o)/float(p); 
	      er = sqrt(ef*(1.-ef)/float(p));
	    }
	    
	    ef=ef*100;
	    er=er*100;
	    
	    //char cam[128];	
	    //sprintf(cam,"%s",rpcsrv.name().c_str());
	    std::string camera = rpcsrv.name();
	    
	    std::cout<<"Integrals for "<<camera<<" is RPC="<<o<<" DT="<<p<<std::endl;

	    float stripsratio = (float(NumberMasked)/float((*r)->nstrips()))*100.;
	    
	    std::cout<<"Strips Masked "<<NumberMasked<<" n Strips="<<(*r)->nstrips()<<std::endl;

	    std::cout<<"Strips Ratio for"<<camera<<" is "<<stripsratio<<std::endl;

	    float mybxhisto = 50.+BXDistribution->getMean()*10;
	    float mybxerror = BXDistribution->getRMS()*10;
	    
	    
	    if((sector==1||sector==2||sector==3||sector==10||sector==11||sector==12)){
	      if(Ring==-2){
		indexWheel[0]++;  
		EffGlobWm2->setBinContent(indexWheel[0],ef);  
		EffGlobWm2->setBinError(indexWheel[0],er);  
		EffGlobWm2->setBinLabel(indexWheel[0],camera,1);

		BXGlobWm2->setBinContent(indexWheel[0],mybxhisto);  
		BXGlobWm2->setBinError(indexWheel[0],mybxerror);  
		//BXGlobWm2->getXaxis()->setBinLabel(indexWheel[0],camera);
	      
		MaskedGlobWm2->setBinContent(indexWheel[0],stripsratio);  
		//MaskedGlobWm2->getXaxis()->setBinLabel(indexWheel[0],camera);
	      }
	    
	      if(Ring==-1){
		indexWheel[1]++;  
		EffGlobWm1->setBinContent(indexWheel[1],ef);  
		EffGlobWm1->setBinError(indexWheel[1],er);  
		EffGlobWm1->setBinLabel(indexWheel[1],camera,1);  
	      
		BXGlobWm1->setBinContent(indexWheel[1],mybxhisto);  
		BXGlobWm1->setBinError(indexWheel[1],mybxerror);  
		//BXGlobWm1->getXaxis()->setBinLabel(indexWheel[1],camera);
	      
		MaskedGlobWm1->setBinContent(indexWheel[1],stripsratio);  
		//MaskedGlobWm1->getXaxis()->setBinLabel(indexWheel[1],camera);
	      }
	    
	      if(Ring==0){
		indexWheel[2]++;  
		EffGlobW0->setBinContent(indexWheel[2],ef);  
		EffGlobW0->setBinError(indexWheel[2],er);  
		EffGlobW0->setBinLabel(indexWheel[2],camera,1);  
	      
		BXGlobW0->setBinContent(indexWheel[2],mybxhisto);  
		BXGlobW0->setBinError(indexWheel[2],mybxerror);  
		//BXGlobW0->getXaxis()->setBinLabel(indexWheel[2],camera);

		MaskedGlobW0->setBinContent(indexWheel[2],stripsratio);  
		//MaskedGlobW0->getXaxis()->setBinLabel(indexWheel[2],camera);
	      }
	
	      if(Ring==1){
		indexWheel[3]++;  
		EffGlobW1->setBinContent(indexWheel[3],ef);  
		EffGlobW1->setBinError(indexWheel[3],er);  
		EffGlobW1->setBinLabel(indexWheel[3],camera,1);  
	      
		BXGlobW1->setBinContent(indexWheel[3],mybxhisto);  
		BXGlobW1->setBinError(indexWheel[3],mybxerror);  
		//BXGlobW1->getXaxis()->setBinLabel(indexWheel[3],camera);

		MaskedGlobW1->setBinContent(indexWheel[3],stripsratio);  
		//MaskedGlobW1->getXaxis()->setBinLabel(indexWheel[3],camera);
	      }
	    
	      if(Ring==2){
		indexWheel[4]++;
		EffGlobW2->setBinContent(indexWheel[4],ef);
		EffGlobW2->setBinError(indexWheel[4],er);
		EffGlobW2->setBinLabel(indexWheel[4],camera,1);

		BXGlobW2->setBinContent(indexWheel[4],mybxhisto);  
		BXGlobW2->setBinError(indexWheel[4],mybxerror);  
		//BXGlobW2->getXaxis()->setBinLabel(indexWheel[4],camera);
	      
		MaskedGlobW2->setBinContent(indexWheel[4],stripsratio);  
		//MaskedGlobW2->getXaxis()->setBinLabel(indexWheel[4],camera);
	      }
	    }else{
	      
	      if(Ring==-2){
		indexWheelf[0]++;  
		EffGlobWm2far->setBinContent(indexWheelf[0],ef);  
		EffGlobWm2far->setBinError(indexWheelf[0],er);  
		EffGlobWm2far->setBinLabel(indexWheelf[0],camera,1);

		BXGlobWm2far->setBinContent(indexWheelf[0],mybxhisto);  
		BXGlobWm2far->setBinError(indexWheelf[0],mybxerror);  
		//BXGlobWm2far->getXaxis()->setBinLabel(indexWheelf[0],camera);
	      
		MaskedGlobWm2far->setBinContent(indexWheelf[0],stripsratio);
		//MaskedGlobWm2far->getXaxis()->setBinLabel(indexWheelf[0],camera);
	      }
	    
	      if(Ring==-1){
		indexWheelf[1]++;  
		EffGlobWm1far->setBinContent(indexWheelf[1],ef);  
		EffGlobWm1far->setBinError(indexWheelf[1],er);  
		EffGlobWm1far->setBinLabel(indexWheelf[1],camera,1);  
	      
		BXGlobWm1far->setBinContent(indexWheelf[1],mybxhisto);  
		BXGlobWm1far->setBinError(indexWheelf[1],mybxerror);  
		//BXGlobWm1far->getXaxis()->setBinLabel(indexWheelf[1],camera);
	      
		MaskedGlobWm1far->setBinContent(indexWheelf[1],stripsratio);
		//MaskedGlobWm1far->getXaxis()->setBinLabel(indexWheelf[1],camera);
	      }
	    
	      if(Ring==0){
		indexWheelf[2]++;  
		EffGlobW0far->setBinContent(indexWheelf[2],ef);  
		EffGlobW0far->setBinError(indexWheelf[2],er);  
		EffGlobW0far->setBinLabel(indexWheelf[2],camera,1);  
	      
		BXGlobW0far->setBinContent(indexWheelf[2],mybxhisto);  
		BXGlobW0far->setBinError(indexWheelf[2],mybxerror);  
		//BXGlobW0far->getXaxis()->setBinLabel(indexWheelf[2],camera);

		MaskedGlobW0far->setBinContent(indexWheelf[2],stripsratio);
		//MaskedGlobW0far->getXaxis()->setBinLabel(indexWheelf[2],camera);
	      }
	
	      if(Ring==1){
		indexWheelf[3]++;  
		EffGlobW1far->setBinContent(indexWheelf[3],ef);  
		EffGlobW1far->setBinError(indexWheelf[3],er);  
		EffGlobW1far->setBinLabel(indexWheelf[3],camera,1);  
	      
		BXGlobW1far->setBinContent(indexWheelf[3],mybxhisto);  
		BXGlobW1far->setBinError(indexWheelf[3],mybxerror);  
		//BXGlobW1far->getXaxis()->setBinLabel(indexWheelf[3],camera);

		MaskedGlobW1far->setBinContent(indexWheelf[3],stripsratio);
		//MaskedGlobW1->getXaxis()->setBinLabel(indexWheelf[3],camera);
	      }
	    
	      if(Ring==2){
		indexWheelf[4]++;
		EffGlobW2far->setBinContent(indexWheelf[4],ef);
		EffGlobW2far->setBinError(indexWheelf[4],er);
		EffGlobW2far->setBinLabel(indexWheelf[4],camera,1);

		BXGlobW2far->setBinContent(indexWheelf[4],mybxhisto);  
		BXGlobW2far->setBinError(indexWheelf[4],mybxerror);  
		//BXGlobW2far->getXaxis()->setBinLabel(indexWheelf[4],camera);
	      
		MaskedGlobW2far->setBinContent(indexWheelf[4],stripsratio);
		//MaskedGlobW2far->getXaxis()->setBinLabel(indexWheelf[4],camera);
	      }
	    }
	  }
	}
      }
    }
  }
 
  EffGlobWm2->setAxisRange(0.,100.,2);
  EffGlobWm1->setAxisRange(0.,100.,2);
  EffGlobW0->setAxisRange(0.,100.,2);
  EffGlobW1->setAxisRange(0.,100.,2);
  EffGlobW2->setAxisRange(0.,100.,2);
  
  EffGlobWm2far->setAxisRange(0.,100.,2);
  EffGlobWm1far->setAxisRange(0.,100.,2);
  EffGlobW0far->setAxisRange(0.,100.,2);
  EffGlobW1far->setAxisRange(0.,100.,2);
  EffGlobW2far->setAxisRange(0.,100.,2);
  

  EffGlobWm2->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);
  
  
  
  EffGlobWm2far->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);
  
  
  
  EffGlobWm1->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);
  
  EffGlobWm1far->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);

  EffGlobW0->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);

  EffGlobW0far->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);
  
  EffGlobW1->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);
  
  EffGlobW1far->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);
  
  EffGlobW2->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);
  
  EffGlobW2far->setAxisTitle("Efficiency (%)/Dead Strips (%)",2);
  
  
}


// ------------ method called once each job just after ending the event loop  ------------
void 
RPCEfficiencySecond::endJob(){

  std::cout<<"Begin End Job"<<std::endl;
  std::cout<<"Saving RootFile"<<std::endl;
  dbe->save("/tmp/carrillo/Second.root");
   
}

