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
#include "TCanvas.h"
#include "TTree.h"
#include "TGaxis.h"
#include <TStyle.h>

//
// class decleration
//

class TFile;


RPCEfficiencySecond::RPCEfficiencySecond(const edm::ParameterSet& iConfig){
   //now do what ever initialization is needed
  file=iConfig.getUntrackedParameter<std::string>("fileName");
  fileOut=iConfig.getUntrackedParameter<std::string>("fileOut");
}


RPCEfficiencySecond::~RPCEfficiencySecond()
{}


void 
RPCEfficiencySecond::beginJob(const edm::EventSetup&)
{
  theFile = new TFile(file.c_str());
  theFileout = new TFile(fileOut.c_str(), "RECREATE");
}


void
RPCEfficiencySecond::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  bool first=false;
  std::cout <<"\t Getting the RPC Geometry"<<std::endl;
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  int CanvaSizeX = 800;
  int CanvaSizeY = 600;

  TCanvas * Ca1;
  TCanvas * Ca2;
  
  Ca1 = new TCanvas("Ca1","Efficiency",CanvaSizeX,CanvaSizeY);
  Ca2 = new TCanvas("Ca2","Global Efficiency",1200,CanvaSizeY);
  
  EffGlobWm2= new TH1F("GlobEfficiencyWheel_-2near"," Efficiency Near Wheel -2 ",105,0.5,105.5);
  EffGlobWm1= new TH1F("GlobEfficiencyWheel_-1near"," Efficiency Near Wheel -1",105,0.5,105.5);
  EffGlobW0 = new TH1F("GlobEfficiencyWheel_0near"," Efficiency Near Wheel 0",105,0.5,105.5);
  EffGlobW1 = new TH1F("GlobEfficiencyWheel_1near"," Efficiency Near Wheel 1",105,0.5,105.5);
  EffGlobW2 = new TH1F("GlobEfficiencyWheel_2near"," Efficiency Near Wheel 2",105,0.5,105.5);

  EffGlobWm2far=new TH1F("GlobEfficiencyWheel_-2far"," Efficiency Far Wheel -2",101,0.5,101.5);
  EffGlobWm1far=new TH1F("GlobEfficiencyWheel_-1far"," Efficiency Far Wheel -1",101,0.5,101.5);
  EffGlobW0far =new TH1F("GlobEfficiencyWheel_0far"," Efficiency Far Wheel 0",101,0.5,101.5);
  EffGlobW1far =new TH1F("GlobEfficiencyWheel_1far"," Efficiency Far Wheel 1",101,0.5,101.5);
  EffGlobW2far =new TH1F("GlobEfficiencyWheel_2far"," Efficiency Far Wheel 2",101,0.5,101.5);

  BXGlobWm2= new TH1F("GlobBXWheel_-2near"," BX Near Wheel -2",105,0.5,105.5);
  BXGlobWm1= new TH1F("GlobBXWheel_-1near"," BX Near Wheel -1",105,0.5,105.5);
  BXGlobW0 = new TH1F("GlobBXWheel_0near"," BX Near Wheel 0",105,0.5,105.5);
  BXGlobW1 = new TH1F("GlobBXWheel_1near"," BX Near Wheel 1",105,0.5,105.5);
  BXGlobW2 = new TH1F("GlobBXWheel_2near"," BX Near Wheel 2",105,0.5,105.5);
  
  BXGlobWm2far= new TH1F("GlobBXWheel_-2far"," BX Far Wheel -2",101,0.5,101.5);
  BXGlobWm1far= new TH1F("GlobBXWheel_-1far"," BX Far Wheel -1",101,0.5,101.5);
  BXGlobW0far = new TH1F("GlobBXWheel_0far"," BX Far Wheel 0",101,0.5,101.5);
  BXGlobW1far = new TH1F("GlobBXWheel_1far"," BX Far Wheel 1",101,0.5,101.5);
  BXGlobW2far = new TH1F("GlobBXWheel_2far"," BX Far Wheel 2",101,0.5,101.5);

  MaskedGlobWm2= new TH1F("GlobMaskedWheel_-2near"," Masked Near Wheel -2",105,0.5,105.5);
  MaskedGlobWm1= new TH1F("GlobMaskedWheel_-1near"," Masked Near Wheel -1",105,0.5,105.5);
  MaskedGlobW0 = new TH1F("GlobMaskedWheel_0near"," Masked Near Wheel 0",105,0.5,105.5);
  MaskedGlobW1 = new TH1F("GlobMaskedWheel_1near"," Masked Near Wheel 1",105,0.5,105.5);
  MaskedGlobW2 = new TH1F("GlobMaskedWheel_2near"," Masked Near Wheel 2",105,0.5,105.5);

  MaskedGlobWm2far= new TH1F("GlobMaskedWheel_-2far"," Masked Far Wheel -2",101,0.5,101.5);
  MaskedGlobWm1far= new TH1F("GlobMaskedWheel_-1far"," Masked Far Wheel -1",101,0.5,101.5);
  MaskedGlobW0far = new TH1F("GlobMaskedWheel_0far"," Masked Far Wheel 0",101,0.5,101.5);
  MaskedGlobW1far = new TH1F("GlobMaskedWheel_1far"," Masked Far Wheel 1",101,0.5,101.5);
  MaskedGlobW2far = new TH1F("GlobMaskedWheel_2far"," Masked Far Wheel 2",101,0.5,101.5);
  
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
	  
	  sprintf(folder,"DQMData/RPC/MuonSegEff/%s/%s_%d/station_%d/sector_%d",
		  regionName.c_str(),ringType.c_str(),rpcId.ring(),rpcId.station(),rpcId.sector());
	  sprintf(detUnitLabel ,"%s",rpcsrv.name().c_str());
	
	  sprintf(meIdRPC,"%s/RPCDataOccupancyFromDT_%s",folder,detUnitLabel);
	
	  sprintf(meIdDT,"%s/ExpectedOccupancyFromDT_%s",folder,detUnitLabel);
	  
	  sprintf(bxDistroId,"%s/BXDistribution_%s",folder,detUnitLabel);
	  sprintf(meIdRealRPC,"%s/RealDetectedOccupancyFromDT_%s",folder,detUnitLabel);  
	  
	  std::cout<<folder<<std::endl;
	  
	  histoRPC= (TH1F*)theFile->Get(meIdRPC);
	  histoDT= (TH1F*)theFile->Get(meIdDT);

	  BXDistribution = (TH1F*)theFile->Get(bxDistroId);
	  histoRealRPC = (TH1F*)theFile->Get(meIdRealRPC);
	  	  
	  std::cout<<"Before If..."<<std::endl;
	  
	  if(histoRPC && histoDT && histoRPC_2D && histoDT_2D && BXDistribution && histoRealRPC){
	    
	    std::cout<<"No empty Histogram"<<std::endl;
	    
	    bool somenthing1D = false;
	    
  	    for(unsigned int i=1;i<=int((*r)->nstrips());++i){
	      if(histoDT->GetBinContent(i) != 0){
		//float eff = histoRPC->GetBinContent(i)/histoDT->GetBinContent(i);
		//float erreff = sqrt(eff*(1-eff)/histoDT->GetBinContent(i));
		somenthing1D = true;
		std::cout<<"Bin Content"<<histoDT->GetBinContent(i)<<std::endl;
	      }
	      if(histoRealRPC->GetBinContent(i)==0) NumberMasked++;
	    }
	    
	    if(first){
	      std::cout<<"cd outputfile folder just first time"<<std::endl;
	      theFileout->cd();
	      first=false;
	    }
	    
	    if(somenthing1D){
	      //histoRPC->Write();
	      //histoDT->Write();
	      //sprintf(namefile1D,"results/Efficiency/profile.%s.png",detUnitLabel);
	      //Ca1->SaveAs(namefile1D);
	      //Ca1->Clear();
	    }
	    
	    if(BXDistribution->Integral()!=0){
	      //sprintf(bxFileName,"results/mydqm/bxDistribution.%s.png",detUnitLabel);
	      //BXDistribution->Draw("");
	      //Ca1->SaveAs(bxFileName);
	      //Ca1->Clear();
	    }
	    
	    
	    //Global Efficiency per Wheel
	    
	    int Ring = rpcId.ring();
	    
	    float ef =0;
	    float er =0;
	    
	    double p=histoDT->Integral();
	    double o=histoRPC->Integral();
	    
	    if(p!=0){
	      ef = float(o)/float(p); 
	      er = sqrt(ef*(1.-ef)/float(p));
	    }
	    
	    ef=ef*100;
	    er=er*100;
	    
	    char cam[128];	
	    sprintf(cam,"%s",rpcsrv.name().c_str());
	    TString camera = (TString)cam;
	    
	    std::cout<<"Integrals for "<<camera<<" is RPC="<<o<<" DT="<<p<<std::endl;

	    float stripsratio = (float(NumberMasked)/float((*r)->nstrips()))*100.;
	    
	    std::cout<<"Strips Masked "<<NumberMasked<<" n Strips="<<(*r)->nstrips()<<std::endl;

	    std::cout<<"Strips Ratio for"<<camera<<" is "<<stripsratio<<std::endl;

	    float mybxhisto = 50.+BXDistribution->GetMean()*10;
	    float mybxerror = BXDistribution->GetRMS()*10;
	    
	    
	    if((sector==1||sector==2||sector==3||sector==10||sector==11||sector==12)){
	      if(Ring==-2){
		indexWheel[0]++;  
		EffGlobWm2->SetBinContent(indexWheel[0],ef);  
		EffGlobWm2->SetBinError(indexWheel[0],er);  
		EffGlobWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera);

		BXGlobWm2->SetBinContent(indexWheel[0],mybxhisto);  
		BXGlobWm2->SetBinError(indexWheel[0],mybxerror);  
		//BXGlobWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera);
	      
		MaskedGlobWm2->SetBinContent(indexWheel[0],stripsratio);  
		//MaskedGlobWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera);
	      }
	    
	      if(Ring==-1){
		indexWheel[1]++;  
		EffGlobWm1->SetBinContent(indexWheel[1],ef);  
		EffGlobWm1->SetBinError(indexWheel[1],er);  
		EffGlobWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera);  
	      
		BXGlobWm1->SetBinContent(indexWheel[1],mybxhisto);  
		BXGlobWm1->SetBinError(indexWheel[1],mybxerror);  
		//BXGlobWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera);
	      
		MaskedGlobWm1->SetBinContent(indexWheel[1],stripsratio);  
		//MaskedGlobWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera);
	      }
	    
	      if(Ring==0){
		indexWheel[2]++;  
		EffGlobW0->SetBinContent(indexWheel[2],ef);  
		EffGlobW0->SetBinError(indexWheel[2],er);  
		EffGlobW0->GetXaxis()->SetBinLabel(indexWheel[2],camera);  
	      
		BXGlobW0->SetBinContent(indexWheel[2],mybxhisto);  
		BXGlobW0->SetBinError(indexWheel[2],mybxerror);  
		//BXGlobW0->GetXaxis()->SetBinLabel(indexWheel[2],camera);

		MaskedGlobW0->SetBinContent(indexWheel[2],stripsratio);  
		//MaskedGlobW0->GetXaxis()->SetBinLabel(indexWheel[2],camera);
	      }
	
	      if(Ring==1){
		indexWheel[3]++;  
		EffGlobW1->SetBinContent(indexWheel[3],ef);  
		EffGlobW1->SetBinError(indexWheel[3],er);  
		EffGlobW1->GetXaxis()->SetBinLabel(indexWheel[3],camera);  
	      
		BXGlobW1->SetBinContent(indexWheel[3],mybxhisto);  
		BXGlobW1->SetBinError(indexWheel[3],mybxerror);  
		//BXGlobW1->GetXaxis()->SetBinLabel(indexWheel[3],camera);

		MaskedGlobW1->SetBinContent(indexWheel[3],stripsratio);  
		//MaskedGlobW1->GetXaxis()->SetBinLabel(indexWheel[3],camera);
	      }
	    
	      if(Ring==2){
		indexWheel[4]++;
		EffGlobW2->SetBinContent(indexWheel[4],ef);
		EffGlobW2->SetBinError(indexWheel[4],er);
		EffGlobW2->GetXaxis()->SetBinLabel(indexWheel[4],camera);

		BXGlobW2->SetBinContent(indexWheel[4],mybxhisto);  
		BXGlobW2->SetBinError(indexWheel[4],mybxerror);  
		//BXGlobW2->GetXaxis()->SetBinLabel(indexWheel[4],camera);
	      
		MaskedGlobW2->SetBinContent(indexWheel[4],stripsratio);  
		//MaskedGlobW2->GetXaxis()->SetBinLabel(indexWheel[4],camera);
	      }
	    }else{
	      
	      if(Ring==-2){
		indexWheelf[0]++;  
		EffGlobWm2far->SetBinContent(indexWheelf[0],ef);  
		EffGlobWm2far->SetBinError(indexWheelf[0],er);  
		EffGlobWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera);

		BXGlobWm2far->SetBinContent(indexWheelf[0],mybxhisto);  
		BXGlobWm2far->SetBinError(indexWheelf[0],mybxerror);  
		//BXGlobWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera);
	      
		MaskedGlobWm2far->SetBinContent(indexWheelf[0],stripsratio);
		//MaskedGlobWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera);
	      }
	    
	      if(Ring==-1){
		indexWheelf[1]++;  
		EffGlobWm1far->SetBinContent(indexWheelf[1],ef);  
		EffGlobWm1far->SetBinError(indexWheelf[1],er);  
		EffGlobWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera);  
	      
		BXGlobWm1far->SetBinContent(indexWheelf[1],mybxhisto);  
		BXGlobWm1far->SetBinError(indexWheelf[1],mybxerror);  
		//BXGlobWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera);
	      
		MaskedGlobWm1far->SetBinContent(indexWheelf[1],stripsratio);
		//MaskedGlobWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera);
	      }
	    
	      if(Ring==0){
		indexWheelf[2]++;  
		EffGlobW0far->SetBinContent(indexWheelf[2],ef);  
		EffGlobW0far->SetBinError(indexWheelf[2],er);  
		EffGlobW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera);  
	      
		BXGlobW0far->SetBinContent(indexWheelf[2],mybxhisto);  
		BXGlobW0far->SetBinError(indexWheelf[2],mybxerror);  
		//BXGlobW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera);

		MaskedGlobW0far->SetBinContent(indexWheelf[2],stripsratio);
		//MaskedGlobW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera);
	      }
	
	      if(Ring==1){
		indexWheelf[3]++;  
		EffGlobW1far->SetBinContent(indexWheelf[3],ef);  
		EffGlobW1far->SetBinError(indexWheelf[3],er);  
		EffGlobW1far->GetXaxis()->SetBinLabel(indexWheelf[3],camera);  
	      
		BXGlobW1far->SetBinContent(indexWheelf[3],mybxhisto);  
		BXGlobW1far->SetBinError(indexWheelf[3],mybxerror);  
		//BXGlobW1far->GetXaxis()->SetBinLabel(indexWheelf[3],camera);

		MaskedGlobW1far->SetBinContent(indexWheelf[3],stripsratio);
		//MaskedGlobW1->GetXaxis()->SetBinLabel(indexWheelf[3],camera);
	      }
	    
	      if(Ring==2){
		indexWheelf[4]++;
		EffGlobW2far->SetBinContent(indexWheelf[4],ef);
		EffGlobW2far->SetBinError(indexWheelf[4],er);
		EffGlobW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera);

		BXGlobW2far->SetBinContent(indexWheelf[4],mybxhisto);  
		BXGlobW2far->SetBinError(indexWheelf[4],mybxerror);  
		//BXGlobW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera);
	      
		MaskedGlobW2far->SetBinContent(indexWheelf[4],stripsratio);
		//MaskedGlobW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera);
	      }
	    }
	  }
	}
      }
    }
  }
 
  EffGlobWm2->GetXaxis()->LabelsOption("v");
  EffGlobWm1->GetXaxis()->LabelsOption("v");
  EffGlobW0->GetXaxis()->LabelsOption("v");
  EffGlobW1->GetXaxis()->LabelsOption("v");
  EffGlobW2->GetXaxis()->LabelsOption("v");

  EffGlobWm2->GetXaxis()->SetLabelSize(0.03);
  EffGlobWm1->GetXaxis()->SetLabelSize(0.03);
  EffGlobW0->GetXaxis()->SetLabelSize(0.03);
  EffGlobW1->GetXaxis()->SetLabelSize(0.03);
  EffGlobW2->GetXaxis()->SetLabelSize(0.03);

  EffGlobWm2->GetYaxis()->SetRangeUser(0.,100.);
  EffGlobWm1->GetYaxis()->SetRangeUser(0.,100.);
  EffGlobW0->GetYaxis()->SetRangeUser(0.,100.);
  EffGlobW1->GetYaxis()->SetRangeUser(0.,100.);
  EffGlobW2->GetYaxis()->SetRangeUser(0.,100.);
  
  BXGlobWm2->GetXaxis()->LabelsOption("v");
  BXGlobWm1->GetXaxis()->LabelsOption("v");
  BXGlobW0->GetXaxis()->LabelsOption("v");
  BXGlobW1->GetXaxis()->LabelsOption("v");
  BXGlobW2->GetXaxis()->LabelsOption("v");

  MaskedGlobWm2->GetXaxis()->LabelsOption("v");
  MaskedGlobWm1->GetXaxis()->LabelsOption("v");
  MaskedGlobW0->GetXaxis()->LabelsOption("v");
  MaskedGlobW1->GetXaxis()->LabelsOption("v");
  MaskedGlobW2->GetXaxis()->LabelsOption("v");


  EffGlobWm2far->GetXaxis()->LabelsOption("v");
  EffGlobWm1far->GetXaxis()->LabelsOption("v");
  EffGlobW0far->GetXaxis()->LabelsOption("v");
  EffGlobW1far->GetXaxis()->LabelsOption("v");
  EffGlobW2far->GetXaxis()->LabelsOption("v");

  EffGlobWm2far->GetXaxis()->SetLabelSize(0.03);
  EffGlobWm1far->GetXaxis()->SetLabelSize(0.03);
  EffGlobW0far->GetXaxis()->SetLabelSize(0.03);
  EffGlobW1far->GetXaxis()->SetLabelSize(0.03);
  EffGlobW2far->GetXaxis()->SetLabelSize(0.03);

  EffGlobWm2far->GetYaxis()->SetRangeUser(0.,100.);
  EffGlobWm1far->GetYaxis()->SetRangeUser(0.,100.);
  EffGlobW0far->GetYaxis()->SetRangeUser(0.,100.);
  EffGlobW1far->GetYaxis()->SetRangeUser(0.,100.);
  EffGlobW2far->GetYaxis()->SetRangeUser(0.,100.);
  

  BXGlobWm2far->GetXaxis()->LabelsOption("v");
  BXGlobWm1far->GetXaxis()->LabelsOption("v");
  BXGlobW0far->GetXaxis()->LabelsOption("v");
  BXGlobW1far->GetXaxis()->LabelsOption("v");
  BXGlobW2far->GetXaxis()->LabelsOption("v");

  MaskedGlobWm2far->GetXaxis()->LabelsOption("v");
  MaskedGlobWm1far->GetXaxis()->LabelsOption("v");
  MaskedGlobW0far->GetXaxis()->LabelsOption("v");
  MaskedGlobW1far->GetXaxis()->LabelsOption("v");
  MaskedGlobW2far->GetXaxis()->LabelsOption("v");

  Ca2->SetBottomMargin(0.4);
  
  TGaxis * bxAxis = new TGaxis(90.,0.,90.,105.,-5,5,11,"+L");
  bxAxis->SetLabelColor(9);
  bxAxis->SetName("bxAxis");
  bxAxis->SetTitle("Mean BX");
  bxAxis->SetTitleColor(9);
  bxAxis->CenterTitle();
  bxAxis->Draw("same");
  gStyle->SetOptStat(0);

  EffGlobWm2->LabelsDeflate();
  EffGlobWm2->Draw();
  EffGlobWm2->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobWm2->LabelsDeflate();
  BXGlobWm2->SetMarkerColor(9);
  BXGlobWm2->SetLineColor(9);
  BXGlobWm2->Draw("same");

  MaskedGlobWm2->LabelsDeflate();
  MaskedGlobWm2->SetMarkerColor(2);
  MaskedGlobWm2->SetLineColor(2);
  MaskedGlobWm2->Draw("same");
  
  Ca2->SaveAs("BxDeadStripEffFromLocalWm2near.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalWm2near.root");
  Ca2->Clear();

  EffGlobWm2far->LabelsDeflate();
  EffGlobWm2far->Draw();
  EffGlobWm2far->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobWm2far->LabelsDeflate();
  BXGlobWm2far->SetMarkerColor(9);
  BXGlobWm2far->SetLineColor(9);
  BXGlobWm2far->Draw("same");

  MaskedGlobWm2far->LabelsDeflate();
  MaskedGlobWm2far->SetMarkerColor(2);
  MaskedGlobWm2far->SetLineColor(2);
  MaskedGlobWm2far->Draw("same");
  
  bxAxis->Draw("same");
  
  Ca2->SaveAs("BxDeadStripEffFromLocalWm2far.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalWm2far.root");
  Ca2->Clear();

  EffGlobWm1->LabelsDeflate();
  EffGlobWm1->Draw();
  EffGlobWm1->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobWm1->LabelsDeflate();
  BXGlobWm1->SetMarkerColor(9);
  BXGlobWm1->SetLineColor(9);
  BXGlobWm1->Draw("same");

  MaskedGlobWm1->LabelsDeflate();
  MaskedGlobWm1->SetMarkerColor(2);
  MaskedGlobWm1->SetLineColor(2);
  MaskedGlobWm1->Draw("same");

  bxAxis->Draw("same");

  Ca2->SaveAs("BxDeadStripEffFromLocalWm1near.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalWm1near.root");
  Ca2->Clear();

  EffGlobWm1far->LabelsDeflate();
  EffGlobWm1far->Draw();
  EffGlobWm1far->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobWm1far->LabelsDeflate();
  BXGlobWm1far->SetMarkerColor(9);
  BXGlobWm1far->SetLineColor(9);
  BXGlobWm1far->Draw("same");

  MaskedGlobWm1far->LabelsDeflate();
  MaskedGlobWm1far->SetMarkerColor(2);
  MaskedGlobWm1far->SetLineColor(2);
  MaskedGlobWm1far->Draw("same");

  bxAxis->Draw("same");

  Ca2->SaveAs("BxDeadStripEffFromLocalWm1far.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalWm1far.root");
  Ca2->Clear();

  EffGlobW0->LabelsDeflate();
  EffGlobW0->Draw();
  EffGlobW0->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobW0->LabelsDeflate();
  BXGlobW0->SetMarkerColor(9);
  BXGlobW0->SetLineColor(9);
  BXGlobW0->Draw("same");

  MaskedGlobW0->LabelsDeflate();
  MaskedGlobW0->SetMarkerColor(2);
  MaskedGlobW0->SetLineColor(2);
  MaskedGlobW0->Draw("same");

  bxAxis->Draw("same");

  Ca2->SaveAs("BxDeadStripEffFromLocalW0near.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalW0near.root");
  Ca2->Clear();

  EffGlobW0far->LabelsDeflate();
  EffGlobW0far->Draw();
  EffGlobW0far->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobW0far->LabelsDeflate();
  BXGlobW0far->SetMarkerColor(9);
  BXGlobW0far->SetLineColor(9);
  BXGlobW0far->Draw("same");

  MaskedGlobW0far->LabelsDeflate();
  MaskedGlobW0far->SetMarkerColor(2);
  MaskedGlobW0far->SetLineColor(2);
  MaskedGlobW0far->Draw("same");

  bxAxis->Draw("same");

  Ca2->SaveAs("BxDeadStripEffFromLocalW0far.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalW0far.root");
  Ca2->Clear();

  EffGlobW1->LabelsDeflate();
  EffGlobW1->Draw();
  EffGlobW1->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobW1->LabelsDeflate();
  BXGlobW1->SetMarkerColor(9);
  BXGlobW1->SetLineColor(9);
  BXGlobW1->Draw("same");

  MaskedGlobW1->LabelsDeflate();
  MaskedGlobW1->SetMarkerColor(2);
  MaskedGlobW1->SetLineColor(2);
  MaskedGlobW1->Draw("same");

  bxAxis->Draw("same");

  Ca2->SaveAs("BxDeadStripEffFromLocalW1near.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalW1near.root");
  Ca2->Clear();

  EffGlobW1far->LabelsDeflate();
  EffGlobW1far->Draw();
  EffGlobW1far->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobW1far->LabelsDeflate();
  BXGlobW1far->SetMarkerColor(9);
  BXGlobW1far->SetLineColor(9);
  BXGlobW1far->Draw("same");

  MaskedGlobW1far->LabelsDeflate();
  MaskedGlobW1far->SetMarkerColor(2);
  MaskedGlobW1far->SetLineColor(2);
  MaskedGlobW1far->Draw("same");

  bxAxis->Draw("same");

  Ca2->SaveAs("BxDeadStripEffFromLocalW1far.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalW1far.root");
  Ca2->Clear();

  EffGlobW2->LabelsDeflate();
  EffGlobW2->Draw();
  EffGlobW2->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobW2->LabelsDeflate();
  BXGlobW2->SetMarkerColor(9);
  BXGlobW2->SetLineColor(9);
  BXGlobW2->Draw("same");

  MaskedGlobW2->LabelsDeflate();
  MaskedGlobW2->SetMarkerColor(2);
  MaskedGlobW2->SetLineColor(2);
  MaskedGlobW2->Draw("same");
  
  bxAxis->Draw("same");

  Ca2->SaveAs("BxDeadStripEffFromLocalW2near.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalW2near.root");
  Ca2->Clear();
  
  EffGlobW2far->LabelsDeflate();
  EffGlobW2far->Draw();
  EffGlobW2far->GetYaxis()->SetTitle("Efficiency (%)/Dead Strips (%)");
  
  BXGlobW2far->LabelsDeflate();
  BXGlobW2far->SetMarkerColor(9);
  BXGlobW2far->SetLineColor(9);
  BXGlobW2far->Draw("same");

  MaskedGlobW2far->LabelsDeflate();
  MaskedGlobW2far->SetMarkerColor(2);
  MaskedGlobW2far->SetLineColor(2);
  MaskedGlobW2far->Draw("same");
  
  bxAxis->Draw("same");
  
  Ca2->SaveAs("BxDeadStripEffFromLocalW2far.png");
  Ca2->SaveAs("BxDeadStripEffFromLocalW2far.root");
  Ca2->Clear();

  theFileout->cd();

  EffGlobWm2->Write();
  EffGlobWm1->Write();
  EffGlobW0->Write();
  EffGlobW1->Write();
  EffGlobW2->Write();

  EffGlobWm2far->Write();
  EffGlobWm1far->Write();
  EffGlobW0far->Write();
  EffGlobW1far->Write();
  EffGlobW2far->Write();

  BXGlobWm2->Write();
  BXGlobWm1->Write();
  BXGlobW0->Write();
  BXGlobW1->Write();
  BXGlobW2->Write();

  BXGlobWm2far->Write();
  BXGlobWm1far->Write();
  BXGlobW0far->Write();
  BXGlobW1far->Write();
  BXGlobW2far->Write();

  MaskedGlobWm2->Write();
  MaskedGlobWm1->Write();
  MaskedGlobW0->Write();
  MaskedGlobW1->Write();
  MaskedGlobW2->Write();

  MaskedGlobWm2far->Write();
  MaskedGlobWm1far->Write();
  MaskedGlobW0far->Write();
  MaskedGlobW1far->Write();
  MaskedGlobW2far->Write();

  Ca2->Close();
  
  Ca1->Close();
  theFileout->Close();
  theFile->Close();
  rollsWithOutData.close();
  rollsWithData.close();
  rollsBarrel.close();
  rollsEndCap.close();
  rpcInfo.close();
  bxMeanList.close();
  
}


// ------------ method called once each job just after ending the event loop  ------------
void 
RPCEfficiencySecond::endJob(){
    
}

