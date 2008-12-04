// -*- C++ -*-
//
// Package:    RPCMonitorEfficiency
// Class:      RPCMonitorEfficiency
// 
/**\class RPCMonitorEfficiency RPCMonitorEfficiency.cc DQM/RPCMonitorDigi/src/RPCMonitorEfficiency.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/45
//         Created:  Tue May 13 12:23:34 CEST 2008
// $Id: RPCMonitorEfficiency.cc,v 1.13 2008/11/28 19:25:40 carrillo Exp $
//
//


// system include files
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
#include<string>
#include<fstream>
#include <DQMOffline/Muon/interface/RPCBookFolderStructure.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

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
#include "TText.h"
#include "TPaveText.h"
//
// class decleration
//

class TFile;

class RPCMonitorEfficiency : public edm::EDAnalyzer {
public:
  explicit RPCMonitorEfficiency(const edm::ParameterSet&);
  ~RPCMonitorEfficiency();  
  TFile * theFile;
  TFile * theFileOut;
  
  TH1F * statistics;

  TH2F * bxbarrel;
  TH2F * bxendcap;
  
  TH1F * hGlobalResClu1La1;
  TH1F * hGlobalResClu1La2;
  TH1F * hGlobalResClu1La3;
  TH1F * hGlobalResClu1La4;
  TH1F * hGlobalResClu1La5;
  TH1F * hGlobalResClu1La6;

  TH1F * hGlobalResClu2La1;
  TH1F * hGlobalResClu2La2;
  TH1F * hGlobalResClu2La3;
  TH1F * hGlobalResClu2La4;
  TH1F * hGlobalResClu2La5;
  TH1F * hGlobalResClu2La6;

  TH1F * hGlobalResClu3La1;
  TH1F * hGlobalResClu3La2;
  TH1F * hGlobalResClu3La3;
  TH1F * hGlobalResClu3La4;
  TH1F * hGlobalResClu3La5;
  TH1F * hGlobalResClu3La6;

  //Endcap

  TH1F * hGlobalResClu1R3C;
  TH1F * hGlobalResClu1R3B;
  TH1F * hGlobalResClu1R3A;
  TH1F * hGlobalResClu1R2C;
  TH1F * hGlobalResClu1R2B;
  TH1F * hGlobalResClu1R2A;

  TH1F * hGlobalResClu2R3C;
  TH1F * hGlobalResClu2R3B;
  TH1F * hGlobalResClu2R3A;
  TH1F * hGlobalResClu2R2C;
  TH1F * hGlobalResClu2R2B;
  TH1F * hGlobalResClu2R2A;

  TH1F * hGlobalResClu3R3C;
  TH1F * hGlobalResClu3R3B;
  TH1F * hGlobalResClu3R3A;
  TH1F * hGlobalResClu3R2C;
  TH1F * hGlobalResClu3R2B;
  TH1F * hGlobalResClu3R2A;
  
  TH1F * EffBarrel;

  TH1F * DoubleGapBarrel;

  TH1F * EffDistroWm2;
  TH1F * EffDistroWm1;
  TH1F * EffDistroW0;
  TH1F * EffDistroW1;
  TH1F * EffDistroW2;

  TH1F * EffDistroWm2far;
  TH1F * EffDistroWm1far;
  TH1F * EffDistroW0far;
  TH1F * EffDistroW1far;
  TH1F * EffDistroW2far;

  TH1F * DoubleGapDistroWm2;
  TH1F * DoubleGapDistroWm1;
  TH1F * DoubleGapDistroW0;
  TH1F * DoubleGapDistroW1;
  TH1F * DoubleGapDistroW2;

  TH1F * DoubleGapDistroWm2far;
  TH1F * DoubleGapDistroWm1far;
  TH1F * DoubleGapDistroW0far;
  TH1F * DoubleGapDistroW1far;
  TH1F * DoubleGapDistroW2far;

  TH1F * EffEndCap;

  TH1F * EffDistroDm3;  
  TH1F * EffDistroDm2;
  TH1F * EffDistroDm1;
  TH1F * EffDistroD1;
  TH1F * EffDistroD2;
  TH1F * EffDistroD3;

  TH1F * EffDistroDm3far;
  TH1F * EffDistroDm2far;
  TH1F * EffDistroDm1far;
  TH1F * EffDistroD1far;
  TH1F * EffDistroD2far;
  TH1F * EffDistroD3far;

  TH2F * Wheelm2Summary;
  TH2F * Wheelm1Summary;
  TH2F * Wheel0Summary;
  TH2F * Wheel1Summary;
  TH2F * Wheel2Summary;

  TH2F * Diskm3Summary;
  TH2F * Diskm2Summary;
  TH2F * Diskm1Summary;
  TH2F * Disk1Summary;
  TH2F * Disk2Summary;
  TH2F * Disk3Summary;
  
  TH1F * histoRPC;
  TH2F * histoRPC_2D;
  TH1F * histoDT;
  TH2F * histoDT_2D;
  TH1F * histoCSC;
  TH2F * histoCSC_2D;
  TH1F * histoPRO;
  TH2F * histoPRO_2D;
  TH1F * histoRES;
  TH1F * BXDistribution;
  TH1F * histoRealRPC;
  TH1F * histoResidual;

  TH1F * EffGlobWm2;
  TH1F * EffGlobWm1;
  TH1F * EffGlobW0;
  TH1F * EffGlobW1;
  TH1F * EffGlobW2;

  TH1F * EffGlobWm2far;
  TH1F * EffGlobWm1far;
  TH1F * EffGlobW0far;
  TH1F * EffGlobW1far;
  TH1F * EffGlobW2far;

  TH1F * DoubleGapWm2;
  TH1F * DoubleGapWm1;
  TH1F * DoubleGapW0;
  TH1F * DoubleGapW1;
  TH1F * DoubleGapW2;

  TH1F * DoubleGapWm2far;
  TH1F * DoubleGapWm1far;
  TH1F * DoubleGapW0far;
  TH1F * DoubleGapW1far;
  TH1F * DoubleGapW2far;

  TH1F * BXGlobWm2;
  TH1F * BXGlobWm1;
  TH1F * BXGlobW0;
  TH1F * BXGlobW1;
  TH1F * BXGlobW2;

  TH1F * BXGlobWm2far;
  TH1F * BXGlobWm1far;
  TH1F * BXGlobW0far;
  TH1F * BXGlobW1far;
  TH1F * BXGlobW2far;

  TH1F * MaskedGlobWm2;
  TH1F * MaskedGlobWm1;
  TH1F * MaskedGlobW0;
  TH1F * MaskedGlobW1;
  TH1F * MaskedGlobW2;

  TH1F * MaskedGlobWm2far;
  TH1F * MaskedGlobWm1far;
  TH1F * MaskedGlobW0far;
  TH1F * MaskedGlobW1far;
  TH1F * MaskedGlobW2far;

  TH1F * AverageEffWm2;
  TH1F * AverageEffWm1;
  TH1F * AverageEffW0;
  TH1F * AverageEffW1;
  TH1F * AverageEffW2;

  TH1F * AverageEffWm2far;
  TH1F * AverageEffWm1far;
  TH1F * AverageEffW0far;
  TH1F * AverageEffW1far;
  TH1F * AverageEffW2far;
  
  TH1F * NoPredictionWm2;
  TH1F * NoPredictionWm1;
  TH1F * NoPredictionW0;
  TH1F * NoPredictionW1;
  TH1F * NoPredictionW2;

  TH1F * NoPredictionWm2far;
  TH1F * NoPredictionWm1far;
  TH1F * NoPredictionW0far;
  TH1F * NoPredictionW1far;
  TH1F * NoPredictionW2far;

  TH1F * EffGlobDm3;
  TH1F * EffGlobDm2;
  TH1F * EffGlobDm1;
  TH1F * EffGlobD1;
  TH1F * EffGlobD2;
  TH1F * EffGlobD3;

  TH1F * EffGlobDm3far;
  TH1F * EffGlobDm2far;
  TH1F * EffGlobDm1far;
  TH1F * EffGlobD1far;
  TH1F * EffGlobD2far;
  TH1F * EffGlobD3far;

  TH1F * BXGlobDm3;
  TH1F * BXGlobDm2;
  TH1F * BXGlobDm1;
  TH1F * BXGlobD1;
  TH1F * BXGlobD2;
  TH1F * BXGlobD3;
  
  TH1F * BXGlobDm3far;
  TH1F * BXGlobDm2far;
  TH1F * BXGlobDm1far;
  TH1F * BXGlobD1far;
  TH1F * BXGlobD2far;
  TH1F * BXGlobD3far;

  TH1F * MaskedGlobDm3;
  TH1F * MaskedGlobDm2;
  TH1F * MaskedGlobDm1;
  TH1F * MaskedGlobD1;
  TH1F * MaskedGlobD2;
  TH1F * MaskedGlobD3;
  
  TH1F * MaskedGlobDm3far;
  TH1F * MaskedGlobDm2far;
  TH1F * MaskedGlobDm1far;
  TH1F * MaskedGlobD1far;
  TH1F * MaskedGlobD2far;
  TH1F * MaskedGlobD3far;

  TH1F * AverageEffDm3;
  TH1F * AverageEffDm2;
  TH1F * AverageEffDm1;
  TH1F * AverageEffD1;
  TH1F * AverageEffD2;
  TH1F * AverageEffD3;

  TH1F * AverageEffDm3far;
  TH1F * AverageEffDm2far;
  TH1F * AverageEffDm1far;
  TH1F * AverageEffD1far;
  TH1F * AverageEffD2far;
  TH1F * AverageEffD3far;

  TH1F * NoPredictionDm3;
  TH1F * NoPredictionDm2;
  TH1F * NoPredictionDm1;
  TH1F * NoPredictionD1;
  TH1F * NoPredictionD2;
  TH1F * NoPredictionD3;

  TH1F * NoPredictionDm3far;
  TH1F * NoPredictionDm2far;
  TH1F * NoPredictionDm1far;
  TH1F * NoPredictionD1far;
  TH1F * NoPredictionD2far;
  TH1F * NoPredictionD3far;

  TPaveText * pave;

private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  std::string file;
  std::string fileout;
  std::ofstream rpcInfo;
  std::ofstream rpcNames;
  std::ofstream rollsWithData;
  std::ofstream rollsWithOutData;
  std::ofstream rollsBarrel;
  std::ofstream rollsEndCap;
  std::ofstream rollsPointedForASegment;
  std::ofstream rollsNotPointedForASegment;
  std::ofstream bxMeanList;
  bool prodimages;
  bool makehtml;
  bool cosmics;
  bool dosD;
  double threshold;
  bool endcap;
  bool barrel; 
};

int rollY(std::string shortname,std::map<int,std::string> rollNames){
  int myy=0;
  for(int i=1;i<22;i++){
    if(rollNames[i].compare(shortname)==0){
      myy=i;
      return myy;
    }
  }
  if(myy==0){
    //std::cout<<"Check your map or your DetId for "<<shortname<<std::endl;
  }
  return myy;
}

RPCMonitorEfficiency::RPCMonitorEfficiency(const edm::ParameterSet& iConfig){
  //now do what ever initialization is needed
  file=iConfig.getUntrackedParameter<std::string>("fileName");
  fileout=iConfig.getUntrackedParameter<std::string>("fileOut");  
  prodimages=iConfig.getUntrackedParameter<bool>("prodimages");
  makehtml=iConfig.getUntrackedParameter<bool>("makehtml");
  cosmics=iConfig.getUntrackedParameter<bool>("cosmics");
  dosD=iConfig.getUntrackedParameter<bool>("dosD");
  threshold=iConfig.getUntrackedParameter<double>("threshold");
  endcap=iConfig.getUntrackedParameter<bool>("endcap");
  barrel=iConfig.getUntrackedParameter<bool>("barrel");
}


RPCMonitorEfficiency::~RPCMonitorEfficiency(){}

void 
RPCMonitorEfficiency::beginJob(const edm::EventSetup&){
  std::cout <<"Begin Job"<<std::endl;
  theFile = new TFile(file.c_str());
  if(!theFile)std::cout<<"The File Doesn't exist"<<std::endl;
  theFileOut = new TFile(fileout.c_str(), "RECREATE");
  /*rpcInfo.open("RPCInfo.txt");
  rpcNames.open("RPCNames.txt");
  rollsWithOutData.open("rollsWithOutData.txt");
  rollsWithData.open("rollsWithData.txt");
  rollsBarrel.open("rollsBarrel.txt");
  rollsEndCap.open("rollsEndCap.txt");
  rollsPointedForASegment.open("rollsPointedForASegment.txt");
  rollsNotPointedForASegment.open("rollsNotPointedForASegment.txt");
  bxMeanList.open("bxMeanList.txt");*/
}


void RPCMonitorEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  //  bool first=false;
  std::cout <<"\t Getting the RPC Geometry"<<std::endl;
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  int CanvaSizeX = 1200;
  int CanvaSizeY = 600;

  TCanvas * Ca0;
  TCanvas * Ca1;
  TCanvas * Ca2;
  TCanvas * Ca3;
  TCanvas * Ca4;
  TCanvas * Ca5;

  bxendcap= new TH2F ("BXEndCap","BX Distribution for the End Cap",51,-5.0,5.0,51,0.,4.);
  bxbarrel= new TH2F ("BXBarrel","BX Distribution for the Barrel",51,-5.0,5.0,51,0.,4.);
  Ca5 = new TCanvas("Ca5","BX by Regions",800,600);
  
  Ca2 = new TCanvas("Ca2","Global Efficiency",CanvaSizeX,CanvaSizeY);

  EffBarrel = new TH1F ("EffBarrel","Efficieny Distribution For All The Barrel",40,0.5,100.5);

  DoubleGapBarrel = new TH1F ("DoubleGapBarrel","Double Gap Efficieny Distribution For All The Barrel",40,0.5,100.5);

  EffDistroWm2= new TH1F ("EffDistroWm2near","Efficieny Distribution For Near Side Wheel -2",20,0.5,100.5);
  EffDistroWm1= new TH1F ("EffDistroWm1near","Efficieny Distribution For Near Side Wheel -1",20,0.5,100.5);
  EffDistroW0= new TH1F ("EffDistroW0near","Efficieny Distribution For Near Side Wheel 0",20,0.5,100.5);
  EffDistroW1= new TH1F ("EffDistroW1near","Efficieny Distribution For Near Side Wheel 1",20,0.5,100.5);
  EffDistroW2= new TH1F ("EffDistroW2near","Efficieny Distribution For Near Side Wheel 2",20,0.5,100.5);

  EffDistroWm2far= new TH1F ("EffDistroWm2far","Efficieny Distribution For Far Side Wheel -2",20,0.5,100.5);
  EffDistroWm1far= new TH1F ("EffDistroWm1far","Efficieny Distribution For Far Side Wheel -1",20,0.5,100.5);
  EffDistroW0far= new TH1F ("EffDistroW0far","Efficieny Distribution For Far Side Wheel 0",20,0.5,100.5);
  EffDistroW1far= new TH1F ("EffDistroW1far","Efficieny Distribution For Far Side Wheel 1",20,0.5,100.5);
  EffDistroW2far= new TH1F ("EffDistroW2far","Efficieny Distribution For Far Side Wheel 2",20,0.5,100.5);
  
  DoubleGapDistroWm2= new TH1F ("DoubleGapDistroWm2near","DoubleGapEfficieny Distribution For Near Side Wheel -2",20,0.5,100.5);
  DoubleGapDistroWm1= new TH1F ("DoubleGapDistroWm1near","DoubleGapEfficieny Distribution For Near Side Wheel -1",20,0.5,100.5);
  DoubleGapDistroW0= new TH1F ("DoubleGapDistroW0near","DoubleGapEfficieny Distribution For Near Side Wheel 0",20,0.5,100.5);
  DoubleGapDistroW1= new TH1F ("DoubleGapDistroW1near","DoubleGapEfficieny Distribution For Near Side Wheel 1",20,0.5,100.5);
  DoubleGapDistroW2= new TH1F ("DoubleGapDistroW2near","DoubleGapEfficieny Distribution For Near Side Wheel 2",20,0.5,100.5);
  
  DoubleGapDistroWm2far= new TH1F ("DoubleGapDistroWm2far","DoubleGapEfficieny Distribution For Far Side Wheel -2",20,0.5,100.5);
  DoubleGapDistroWm1far= new TH1F ("DoubleGapDistroWm1far","DoubleGapEfficieny Distribution For Far Side Wheel -1",20,0.5,100.5);
  DoubleGapDistroW0far= new TH1F ("DoubleGapDistroW0far","DoubleGapEfficieny Distribution For Far Side Wheel 0",20,0.5,100.5);
  DoubleGapDistroW1far= new TH1F ("DoubleGapDistroW1far","DoubleGapEfficieny Distribution For Far Side Wheel 1",20,0.5,100.5);
  DoubleGapDistroW2far= new TH1F ("DoubleGapDistroW2far","DoubleGapEfficieny Distribution For Far Side Wheel 2",20,0.5,100.5);

  EffEndCap= new TH1F ("EffDistroEndCap ","Efficieny Distribution For All The EndCaps",60,0.5,100.5);

  EffDistroDm3= new TH1F ("EffDistroDm3near","Efficieny Distribution For Near Side Disk -3",20,0.5,100.5);  
  EffDistroDm2= new TH1F ("EffDistroDm2near","Efficieny Distribution For Near Side Disk -2",20,0.5,100.5);
  EffDistroDm1= new TH1F ("EffDistroDm1near","Efficieny Distribution For Near Side Disk -1",20,0.5,100.5);
  EffDistroD1= new TH1F ("EffDistroD1near","Efficieny Distribution For Near Side Disk 1",20,0.5,100.5);
  EffDistroD2= new TH1F ("EffDistroD2near","Efficieny Distribution For Near Side Disk 2",20,0.5,100.5);
  EffDistroD3= new TH1F ("EffDistroD3near","Efficieny Distribution For Near Side Disk 3",20,0.5,100.5);

  EffDistroDm3far= new TH1F ("EffDistroDm3far","Efficieny Distribution For Far Side Disk -3",20,0.5,100.5);
  EffDistroDm2far= new TH1F ("EffDistroDm2far","Efficieny Distribution For Far Side Disk -2",20,0.5,100.5);
  EffDistroDm1far= new TH1F ("EffDistroDm1far","Efficieny Distribution For Far Side Disk -1",20,0.5,100.5);
  EffDistroD1far= new TH1F ("EffDistroD1far","Efficieny Distribution For Far Side Disk 1",20,0.5,100.5);
  EffDistroD2far= new TH1F ("EffDistroD2far","Efficieny Distribution For Far Side Disk 2",20,0.5,100.5);
  EffDistroD3far= new TH1F ("EffDistroD3far","Efficieny Distribution For Far Side Disk 3",20,0.5,100.5);

  DoubleGapWm2= new TH1F ("DoubleGapEffWheel_-2near","Double Gap Efficiency Near Side Wheel -2",101,0.5,101.5);
  DoubleGapWm2far= new TH1F("DoubleGapEffWheel_-2far","Double Gap Efficiency Far Side Wheel -2",105,0.5,105.5);
  EffGlobWm2= new TH1F ("GlobEfficiencyWheel_-2near","Efficiency Near Side Wheel -2",101,0.5,101.5);
  EffGlobWm2far= new TH1F ("GlobEfficiencyWheel_-2far","Efficiency Far Side Wheel -2",105,0.5,105.5);
  BXGlobWm2=  new TH1F ("GlobBXWheel_-2near","BX Near Side Wheel -2",101,0.5,101.5);
  BXGlobWm2far=  new TH1F ("GlobBXWheel_-2far","BX Far Side Wheel -2",105,0.5,105.5);
  MaskedGlobWm2=  new TH1F ("GlobMaskedWheel_-2near","Masked Near Side Wheel -2",101,0.5,101.5);
  MaskedGlobWm2far=  new TH1F ("GlobMaskedWheel_-2far","Masked Far Side Wheel -2",105,0.5,105.5);
  AverageEffWm2= new TH1F ("AverageEfficiencyWheel_-2near","Average Efficiency Near Side Wheel -2 ",101,0.5,101.5);
  AverageEffWm2far = new TH1F ("AverageEfficiencyWheel_-2far","Average Efficiency Far Side Wheel -2 ",105,0.5,105.5);
  NoPredictionWm2= new TH1F ("NoPredictionWheel_-2near","No Predictions Near Side Wheel -2 ",101,0.5,101.5);
  NoPredictionWm2far= new TH1F ("NoPredictionWheel_-2far","No Predictions Efficiency Far Side Wheel -2 ",105,0.5,105.5);
  
  DoubleGapWm1=  new TH1F ("DoubleGapEffWheel_-1near","Double Gap Efficiency Near Side Wheel -1",101,0.5,101.5);
  DoubleGapWm1far= new TH1F ("DoubleGapEffWheel_-1far","Double Gap Efficiency Far Side Wheel -1",105,0.5,105.5);
  EffGlobWm1=  new TH1F ("GlobEfficiencyWheel_-1near","Efficiency Near Side Wheel -1",101,0.5,101.5);
  EffGlobWm1far= new TH1F ("GlobEfficiencyWheel_-1far","Efficiency Far Side Wheel -1",105,0.5,105.5);
  BXGlobWm1=  new TH1F ("GlobBXWheel_-1near","BX Near Side Wheel -1",101,0.5,101.5);
  BXGlobWm1far=  new TH1F ("GlobBXWheel_-1far","BX Far Side Wheel -1",105,0.5,105.5);
  MaskedGlobWm1=  new TH1F ("GlobMaskedWheel_-1near","Masked Near Side Wheel -1",101,0.5,101.5);
  MaskedGlobWm1far=  new TH1F ("GlobMaskedWheel_-1far","Masked Far Side Wheel -1",105,0.5,105.5);
  AverageEffWm1= new TH1F ("AverageEfficiencyWheel_-1near","Average Efficiency Near Side Wheel -1 ",101,0.5,101.5);
  AverageEffWm1far= new TH1F ("AverageEfficiencyWheel_-1far","Average Efficiency Far Side Wheel -1 ",105,0.5,105.5);
  NoPredictionWm1= new TH1F ("NoPredictionWheel_-1near","No Predictions Near Side Wheel -1 ",101,0.5,101.5);
  NoPredictionWm1far= new TH1F ("NoPredictionWheel_-1far","No Predictions Efficiency Far Side Wheel -1 ",105,0.5,105.5);

  DoubleGapW0 =  new TH1F ("DoubleGapEffWheel_0near","Double Gap Efficiency Near Side Wheel 0",101,0.5,101.5);
  DoubleGapW0far = new TH1F ("DoubleGapEffWheel_0far","Double Gap Efficiency Far Side Wheel 0",105,0.5,105.5);
  EffGlobW0 =  new TH1F ("GlobEfficiencyWheel_0near","Efficiency Near Side Wheel 0",101,0.5,101.5);
  EffGlobW0far = new TH1F ("GlobEfficiencyWheel_0far","Efficiency Far Side Wheel 0",105,0.5,105.5);
  BXGlobW0 =  new TH1F ("GlobBXWheel_0near","BX Near Side Wheel 0",101,0.5,101.5);
  BXGlobW0far =  new TH1F ("GlobBXWheel_0far","BX Far Side Wheel 0",105,0.5,105.5);
  MaskedGlobW0 =  new TH1F ("GlobMaskedWheel_0near","Masked Near Side Wheel 0",101,0.5,101.5);
  MaskedGlobW0far =  new TH1F ("GlobMaskedWheel_0far","Masked Far Side Wheel 0",105,0.5,105.5);
  AverageEffW0= new TH1F ("AverageEfficiencyWheel_0near","Average Efficiency Near Side Wheel 0 ",101,0.5,101.5);
  AverageEffW0far= new TH1F ("AverageEfficiencyWheel_0far","Average Efficiency Far Side Wheel 0 ",105,0.5,105.5);
  NoPredictionW0= new TH1F ("NoPredictionWheel_0near","No Predictions Near Side Wheel 0 ",101,0.5,101.5);
  NoPredictionW0far= new TH1F ("NoPredictionWheel_0far","No Predictions Efficiency Far Side Wheel 0 ",105,0.5,105.5);

  DoubleGapW1 =  new TH1F ("DoubleGapEffWheel_1near","Double Gap Efficiency Near Side Wheel 1",101,0.5,101.5);
  DoubleGapW1far = new TH1F ("DoubleGapEffWheel_1far","Double Gap Efficiency Far Side Wheel 1",105,0.5,105.5);  
  EffGlobW1 =  new TH1F ("GlobEfficiencyWheel_1near","Efficiency Near Side Wheel 1",101,0.5,101.5);
  EffGlobW1far = new TH1F ("GlobEfficiencyWheel_1far","Efficiency Far Side Wheel 1",105,0.5,105.5);  
  BXGlobW1 =  new TH1F ("GlobBXWheel_1near","BX Near Side Wheel 1",101,0.5,101.5);
  BXGlobW1far =  new TH1F ("GlobBXWheel_1far","BX Far Side Wheel 1",105,0.5,105.5);
  MaskedGlobW1 =  new TH1F ("GlobMaskedWheel_1near","Masked Near Side Wheel 1",101,0.5,101.5);
  MaskedGlobW1far =  new TH1F ("GlobMaskedWheel_1far","Masked Far Side Wheel 1",105,0.5,105.5);
  AverageEffW1= new TH1F ("AverageEfficiencyWheel_1near","Average Efficiency Near Side Wheel 1 ",101,0.5,101.5);
  AverageEffW1far= new TH1F ("AverageEfficiencyWheel_1far","Average Efficiency Far Side Wheel 1 ",105,0.5,105.5);
  NoPredictionW1= new TH1F ("NoPredictionWheel_1near","No Predictions Near Side Wheel 1 ",101,0.5,101.5);
  NoPredictionW1far= new TH1F ("NoPredictionWheel_1far","No Predictions Efficiency Far Side Wheel 1 ",105,0.5,105.5);

  DoubleGapW2 =  new TH1F ("DoubleGapEffWheel_2near","Double Gap Efficiency Near Side Wheel 2",101,0.5,101.5);
  DoubleGapW2far = new TH1F ("DoubleGapEffWheel_2far","Double Gap Efficiency Far Side Wheel 2",105,0.5,105.5);
  EffGlobW2 =  new TH1F ("GlobEfficiencyWheel_2near","Efficiency Near Side Wheel 2",101,0.5,101.5);
  EffGlobW2far = new TH1F ("GlobEfficiencyWheel_2far","Efficiency Far Side Wheel 2",105,0.5,105.5);
  BXGlobW2 =  new TH1F ("GlobBXWheel_2near","BX Near Side Wheel 2",101,0.5,101.5);
  BXGlobW2far =  new TH1F ("GlobBXWheel_2far","BX Far Side Wheel 2",105,0.5,105.5);
  MaskedGlobW2 =  new TH1F ("GlobMaskedWheel_2near","Masked Near Side Wheel 2",101,0.5,101.5);
  MaskedGlobW2far =  new TH1F ("GlobMaskedWheel_2far","Masked Far Side Wheel 2",105,0.5,105.5);
  AverageEffW2= new TH1F ("AverageEfficiencyWheel_2near","Average Efficiency Near Side Wheel 2 ",101,0.5,101.5);
  AverageEffW2far= new TH1F ("AverageEfficiencyWheel_2far","Average Efficiency Far Side Wheel 2 ",105,0.5,105.5);
  NoPredictionW2= new TH1F ("NoPredictionWheel_2near","No Predictions Near Side Wheel 2 ",101,0.5,101.5);
  NoPredictionW2far= new TH1F ("NoPredictionWheel_2far","No Predictions Efficiency Far Side Wheel 2 ",105,0.5,105.5);
  
  //EndCap

  EffGlobD3 = new TH1F ("GlobEfficiencyDisk_3near","Efficiency Near Side Disk 3",109,0.5,109.5);
  EffGlobD3far =new TH1F ("GlobEfficiencyDisk_3far","Efficiency Far Side Disk 3",109,0.5,109.5);
  BXGlobD3 = new TH1F ("GlobBXDisk_3near","BX Near Side Disk 3",109,0.5,109.5);
  BXGlobD3far = new TH1F ("GlobBXDisk_3far","BX Far Side Disk 3",109,0.5,109.5);
  MaskedGlobD3 = new TH1F ("GlobMaskedDisk_3near","Masked Near Side Disk 3",109,0.5,109.5);
  MaskedGlobD3far = new TH1F ("GlobMaskedDisk_3far","Masked Far Side Disk 3",109,0.5,109.5);
  AverageEffD3=new TH1F ("AverageEfficiencyDisk_3near","Average Efficiency Near Side Disk 3 ",109,0.5,109.5);
  AverageEffD3far=new TH1F ("AverageEfficiencyDisk_3far","Average Efficiency Far Side Disk 3 ",109,0.5,109.5);
  NoPredictionD3=new TH1F ("NoPredictionDisk_3near","No Predictions Near Side Disk 3 ",109,0.5,109.5);
  NoPredictionD3far=new TH1F ("NoPredictionDisk_3far","No Predictions Efficiency Far Side Disk 3 ",109,0.5,109.5);

  EffGlobD2 = new TH1F ("GlobEfficiencyDisk_2near","Efficiency Near Side Disk 2",109,0.5,109.5);
  EffGlobD2far =new TH1F ("GlobEfficiencyDisk_2far","Efficiency Far Side Disk 2",109,0.5,109.5);
  BXGlobD2 = new TH1F ("GlobBXDisk_2near","BX Near Side Disk 2",109,0.5,109.5);
  BXGlobD2far = new TH1F ("GlobBXDisk_2far","BX Far Side Disk 2",109,0.5,109.5);
  MaskedGlobD2 = new TH1F ("GlobMaskedDisk_2near","Masked Near Side Disk 2",109,0.5,109.5);
  MaskedGlobD2far = new TH1F ("GlobMaskedDisk_2far","Masked Far Side Disk 2",109,0.5,109.5);
  AverageEffD2=new TH1F ("AverageEfficiencyDisk_2near","Average Efficiency Near Side Disk 2 ",109,0.5,109.5);
  AverageEffD2far=new TH1F ("AverageEfficiencyDisk_2far","Average Efficiency Far Side Disk 2 ",109,0.5,109.5);
  NoPredictionD2=new TH1F ("NoPredictionDisk_2near","No Predictions Near Side Disk 2 ",109,0.5,109.5);
  NoPredictionD2far=new TH1F ("NoPredictionDisk_2far","No Predictions Efficiency Far Side Disk 2 ",109,0.5,109.5);

  EffGlobD1 = new TH1F ("GlobEfficiencyDisk_1near","Efficiency Near Side Disk 1",109,0.5,109.5);
  EffGlobD1far =new TH1F ("GlobEfficiencyDisk_1far","Efficiency Far Side Disk 1",109,0.5,109.5);
  BXGlobD1 = new TH1F ("GlobBXDisk_1near","BX Near Side Disk 1",109,0.5,109.5);
  BXGlobD1far = new TH1F ("GlobBXDisk_1far","BX Far Side Disk 1",109,0.5,109.5);
  MaskedGlobD1 = new TH1F ("GlobMaskedDisk_1near","Masked Near Side Disk 1",109,0.5,109.5);
  MaskedGlobD1far = new TH1F ("GlobMaskedDisk_1far","Masked Far Side Disk 1",109,0.5,109.5);
  AverageEffD1=new TH1F ("AverageEfficiencyDisk_1near","Average Efficiency Near Side Disk 1 ",109,0.5,109.5);
  AverageEffD1far=new TH1F ("AverageEfficiencyDisk_1far","Average Efficiency Far Side Disk 1 ",109,0.5,109.5);
  NoPredictionD1=new TH1F ("NoPredictionDisk_1near","No Predictions Near Side Disk 1 ",109,0.5,109.5);
  NoPredictionD1far=new TH1F ("NoPredictionDisk_1far","No Predictions Efficiency Far Side Disk 1 ",109,0.5,109.5);

  EffGlobDm1 = new TH1F ("GlobEfficiencyDisk_m1near","Efficiency Near Side Disk -1",109,0.5,109.5);
  EffGlobDm1far =new TH1F ("GlobEfficiencyDisk_m1far","Efficiency Far Side Disk -1",109,0.5,109.5);
  BXGlobDm1 = new TH1F ("GlobBXDisk_m1near","BX Near Side Disk -1",109,0.5,109.5);
  BXGlobDm1far = new TH1F ("GlobBXDisk_m1far","BX Far Side Disk -1",109,0.5,109.5);
  MaskedGlobDm1 = new TH1F ("GlobMaskedDisk_m1near","Masked Near Side Disk -1",109,0.5,109.5);
  MaskedGlobDm1far = new TH1F ("GlobMaskedDisk_m1far","Masked Far Side Disk -1",109,0.5,109.5);
  AverageEffDm1=new TH1F ("AverageEfficiencyDisk_m1near","Average Efficiency Near Side Disk -1 ",109,0.5,109.5);
  AverageEffDm1far=new TH1F ("AverageEfficiencyDisk_m1far","Average Efficiency Far Side Disk -1 ",109,0.5,109.5);
  NoPredictionDm1=new TH1F ("NoPredictionDisk_m1near","No Predictions Near Side Disk -1 ",109,0.5,109.5);
  NoPredictionDm1far=new TH1F ("NoPredictionDisk_m1far","No Predictions Efficiency Far Side Disk -1 ",109,0.5,109.5);

  EffGlobDm2 = new TH1F ("GlobEfficiencyDisk_m2near","Efficiency Near Side Disk -2",109,0.5,109.5);
  EffGlobDm2far =new TH1F ("GlobEfficiencyDisk_m2far","Efficiency Far Side Disk -2",109,0.5,109.5);
  BXGlobDm2 = new TH1F ("GlobBXDisk_m2near","BX Near Side Disk -2",109,0.5,109.5);
  BXGlobDm2far = new TH1F ("GlobBXDisk_m2far","BX Far Side Disk -2",109,0.5,109.5);
  MaskedGlobDm2 = new TH1F ("GlobMaskedDisk_m2near","Masked Near Side Disk -2",109,0.5,109.5);
  MaskedGlobDm2far = new TH1F ("GlobMaskedDisk_m2far","Masked Far Side Disk -2",109,0.5,109.5);
  AverageEffDm2=new TH1F ("AverageEfficiencyDisk_m2near","Average Efficiency Near Side Disk -2 ",109,0.5,109.5);
  AverageEffDm2far=new TH1F ("AverageEfficiencyDisk_m2far","Average Efficiency Far Side Disk -2 ",109,0.5,109.5);
  NoPredictionDm2=new TH1F ("NoPredictionDisk_m2near","No Predictions Near Side Disk -2 ",109,0.5,109.5);
  NoPredictionDm2far=new TH1F ("NoPredictionDisk_m2far","No Predictions Efficiency Far Side Disk -2 ",109,0.5,109.5);

  EffGlobDm3 = new TH1F ("GlobEfficiencyDisk_m3near","Efficiency Near Side Disk -3",109,0.5,109.5);
  EffGlobDm3far =new TH1F ("GlobEfficiencyDisk_m3far","Efficiency Far Side Disk -3",109,0.5,109.5);
  BXGlobDm3 = new TH1F ("GlobBXDisk_m3near","BX Near Side Disk -3",109,0.5,109.5);
  BXGlobDm3far = new TH1F ("GlobBXDisk_m3far","BX Far Side Disk -3",109,0.5,109.5);
  MaskedGlobDm3 = new TH1F ("GlobMaskedDisk_m3near","Masked Near Side Disk -3",109,0.5,109.5);
  MaskedGlobDm3far = new TH1F ("GlobMaskedDisk_m3far","Masked Far Side Disk -3",109,0.5,109.5);
  AverageEffDm3=new TH1F ("AverageEfficiencyDisk_m3near","Average Efficiency Near Side Disk -3 ",109,0.5,109.5);
  AverageEffDm3far=new TH1F ("AverageEfficiencyDisk_m3far","Average Efficiency Far Side Disk -3 ",109,0.5,109.5);
  NoPredictionDm3=new TH1F ("NoPredictionDisk_m3near","No Predictions Near Side Disk -3 ",109,0.5,109.5);
  NoPredictionDm3far=new TH1F ("NoPredictionDisk_m3far","No Predictions Efficiency Far Side Disk -3 ",109,0.5,109.5);

  //Summary Histograms
  
  std::string os;
  os="Efficiency_Roll_vs_Sector_Wheel_-2";                                      
  Wheelm2Summary = new TH2F (os.c_str(), os.c_str(), 12, 0.5,12.5, 22, 0.5, 22.5);
  os="Efficiency_Roll_vs_Sector_Wheel_-1";                                      
  Wheelm1Summary = new TH2F (os.c_str(), os.c_str(), 12, 0.5,12.5, 22, 0.5, 22.5);
  os="Efficiency_Roll_vs_Sector_Wheel_0";                                      
  Wheel0Summary = new TH2F (os.c_str(), os.c_str(), 12, 0.5,12.5, 22, 0.5, 22.5);
  os="Efficiency_Roll_vs_Sector_Wheel_+1";                                      
  Wheel1Summary = new TH2F (os.c_str(), os.c_str(), 12, 0.5,12.5, 22, 0.5, 22.5);
  os="Efficiency_Roll_vs_Sector_Wheel_+2";                                      
  Wheel2Summary = new TH2F (os.c_str(), os.c_str(), 12, 0.5,12.5, 22, 0.5, 22.5);
  
  os="Efficiency_Roll_vs_Sector_Disk_-3";                                      
  Diskm3Summary = new TH2F (os.c_str(), os.c_str(), 6, 0.5,6.5, 12, 0.5, 12.5);
  os="Efficiency_Roll_vs_Sector_Disk_-2";                                      
  Diskm2Summary = new TH2F (os.c_str(), os.c_str(), 6, 0.5,6.5, 12, 0.5, 12.5);
  os="Efficiency_Roll_vs_Sector_Disk_-1";                                      
  Diskm1Summary = new TH2F (os.c_str(), os.c_str(), 6, 0.5,6.5, 12, 0.5, 12.5);
  os="Efficiency_Roll_vs_Sector_Disk_+1";                                      
  Disk1Summary = new TH2F (os.c_str(), os.c_str(), 6, 0.5,6.5, 12, 0.5, 12.5);
  os="Efficiency_Roll_vs_Sector_Disk_+2";                                      
  Disk2Summary = new TH2F (os.c_str(), os.c_str(), 6, 0.5,6.5, 12, 0.5, 12.5);
  os="Efficiency_Roll_vs_Sector_Disk_+3";                                      
  Disk3Summary = new TH2F (os.c_str(), os.c_str(), 6, 0.5,6.5, 12, 0.5, 12.5);
  
  
  //Producing plots for residuals and global statistics.

  Ca3 = new TCanvas("Ca3","Profile",1200,600);
  
  gStyle->SetOptStat(0);
  
  std::string meIdRES,folder,labeltoSave,command;
  
  folder = "DQMData/Muons/MuonSegEff/";
  
  meIdRES = folder + "Statistics";
  statistics = (TH1F*)theFile->Get(meIdRES.c_str());
  statistics->GetXaxis()->LabelsOption("v");
  statistics->GetXaxis()->SetLabelSize(0.035);
  statistics->Draw();
  labeltoSave = "Statistics.png";
  Ca3->SetBottomMargin(0.35);
  Ca3->SaveAs(labeltoSave.c_str()); 
  Ca3->Clear();

  folder = "DQMData/Muons/MuonSegEff/Residuals/Barrel/";
 
  Ca4 = new TCanvas("Ca4","Residuals",800,600);
  
  
  command = "mkdir resBarrel"; system(command.c_str());
  
  meIdRES = folder + "GlobalResidualsClu1La1"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu1La1.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1La2"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu1La2.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1La3"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu1La3.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1La4"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu1La4.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1La5"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu1La5.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1La6"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu1La6.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();

  meIdRES = folder + "GlobalResidualsClu2La1"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu2La1.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2La2"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu2La2.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2La3"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu2La3.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2La4"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu2La4.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2La5"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu2La5.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2La6"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu2La6.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();

  meIdRES = folder + "GlobalResidualsClu3La1"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu3La1.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3La2"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu3La2.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3La3"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu3La3.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3La4"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu3La4.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3La5"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu3La5.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3La6"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resBarrel/ResidualsClu3La6.png"; histoRES->GetXaxis()->SetTitle("(cm)");
  Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  
  folder = "DQMData/Muons/MuonSegEff/Residuals/EndCap/";
 
  command = "mkdir resEndCap"; system(command.c_str());
  
  meIdRES = folder + "GlobalResidualsClu1R2A"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu1R2A.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1R2B"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu1R2B.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1R2C"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu1R2C.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1R3A"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu1R3A.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1R3B"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu1R3B.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu1R3C"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu1R3C.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2R2A"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu2R2A.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2R2B"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu2R2B.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2R2C"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu2R2C.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2R3A"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu2R3A.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2R3B"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu2R3B.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu2R3C"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu2R3C.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3R2A"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu3R2A.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3R2B"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu3R2B.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3R2C"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu3R2C.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3R3A"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu3R3A.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3R3B"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu3R3B.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  meIdRES = folder + "GlobalResidualsClu3R3C"; histoRES = (TH1F*)theFile->Get(meIdRES.c_str());  histoRES->Draw(); labeltoSave = "resEndCap/ResidualsClu3R3C.png"; histoRES->GetXaxis()->SetTitle("(cm)");    Ca4->SetLogy(); Ca4->SaveAs(labeltoSave.c_str()); Ca4->Clear();
  
  //Setting Labels in Summary Label Barrel.
  
  std::stringstream binLabel;

  for(int i=1;i<=12;i++){
    binLabel.str("");
    binLabel<<"Sec "<<i;
    Wheelm2Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Wheelm1Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Wheel0Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Wheel1Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Wheel2Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
  }
  
  std::map<int,std::string> rollNamesInter;
  
  rollNamesInter[1]="RB1in_B";
  rollNamesInter[2]="RB1in_F";
  rollNamesInter[3]="RB1out_B";
  rollNamesInter[4]="RB1out_F";
  rollNamesInter[5]="RB2in_B";
  rollNamesInter[6]="RB2in_M";
  rollNamesInter[7]="RB2in_F";
  rollNamesInter[8]="RB2out_B";
  rollNamesInter[9]="RB2out_F";
  rollNamesInter[10]="RB3-_B";
  rollNamesInter[11]="RB3-_F";
  rollNamesInter[12]="RB3+_B";
  rollNamesInter[13]="RB3+_F";
  rollNamesInter[14]="RB4,-,--_B";
  rollNamesInter[15]="RB4,-,--_F";
  rollNamesInter[16]="RB4,+,++_B";
  rollNamesInter[17]="RB4,+,++_F";
  rollNamesInter[18]="RB4-+_B";
  rollNamesInter[19]="RB4-+_F";
  rollNamesInter[20]="RB4+-_B";
  rollNamesInter[21]="RB4+-_F";

  std::map<int,std::string> rollNamesExter;
  
  for(int i=1;i<=22;i++){
    rollNamesExter[i]=rollNamesInter[i];
    //std::cout<<rollNamesInter[i]<<std::endl;
  }
  
  rollNamesExter[6]="RB2in_F";
  rollNamesExter[7]="RB2out_B";
  rollNamesExter[8]="RB2out_M";
  
  for(int i=1;i<22;i++){
    Wheelm1Summary->GetYaxis()->SetBinLabel(i,rollNamesInter[i].c_str());
    Wheel0Summary->GetYaxis()->SetBinLabel(i,rollNamesInter[i].c_str());
    Wheel1Summary->GetYaxis()->SetBinLabel(i,rollNamesInter[i].c_str());
  }

  for(int i=1;i<22;i++){
    Wheelm2Summary->GetYaxis()->SetBinLabel(i,rollNamesExter[i].c_str());
    Wheel2Summary->GetYaxis()->SetBinLabel(i,rollNamesExter[i].c_str());
  }
  
  //Setting Labels in Summary Label Barrel.

  for(int i=1;i<=6;i++){
    binLabel.str("");
    binLabel<<"Sec "<<i;
    //std::cout<<"Labeling EndCaps"<<binLabel.str()<<std::endl;
    Diskm3Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Diskm2Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Diskm1Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Disk1Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Disk2Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
    Disk3Summary->GetXaxis()->SetBinLabel(i,binLabel.str().c_str());
  }

  for(int ri=2;ri<=3;ri++){
    for(int su=1;su<=6;su++){
      binLabel.str("");
      binLabel<<"Ri"<<ri<<"_Su"<<su;
      //std::cout<<"Labeling EndCaps "<<binLabel.str()<<std::endl;
      Diskm3Summary->GetYaxis()->SetBinLabel((ri-2)*6+su,binLabel.str().c_str());
      Diskm2Summary->GetYaxis()->SetBinLabel((ri-2)*6+su,binLabel.str().c_str());
      Diskm1Summary->GetYaxis()->SetBinLabel((ri-2)*6+su,binLabel.str().c_str());
      Disk1Summary->GetYaxis()->SetBinLabel((ri-2)*6+su,binLabel.str().c_str());
      Disk2Summary->GetYaxis()->SetBinLabel((ri-2)*6+su,binLabel.str().c_str());
      Disk3Summary->GetYaxis()->SetBinLabel((ri-2)*6+su,binLabel.str().c_str());
    }
  }
  
  //exit(1);
  
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


  if(makehtml){

    command = "rm *.html" ; system(command.c_str());

    command = "cat htmltemplates/indexhead.html > indexDm3near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexDm2near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexDm1near.html"; system(command.c_str());

    command = "cat htmltemplates/indexhead.html > indexDm3far.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexDm2far.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexDm1far.html"; system(command.c_str());

    command = "cat htmltemplates/indexhead.html > indexD3near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexD2near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexD1near.html"; system(command.c_str());

    command = "cat htmltemplates/indexhead.html > indexD3far.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexD2far.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexD1far.html"; system(command.c_str());

    command = "cat htmltemplates/indexhead.html > indexWm2near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexWm2far.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexWm1near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexWm1far.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexW0near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexW0far.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexW1near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexW1far.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexW2near.html"; system(command.c_str());
    command = "cat htmltemplates/indexhead.html > indexW2far.html"; system(command.c_str());
    
  }
  
  //std::cout<<"Before Rolls Loop"<<std::endl;
  
  Ca0 = new TCanvas("Ca0","Profile",400,300);
  
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	RPCGeomServ rpcsrv(rpcId);
	
	int sector = rpcId.sector();
	int station = rpcId.station();

	int nstrips = int((*r)->nstrips());

	if(rpcId.region()==0 && barrel && (!cosmics||((sector!=1||sector!=7) && station!=4))){  
	  
	  const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&((*r)->topology()));
	  float stripl = top_->stripLength();
	  float stripw = top_->pitch();
	     
	  std::string detUnitLabel, meIdRPC, meIdRPC_2D, meIdDT, meIdDT_2D, meIdPRO, meIdPRO_2D, bxDistroId, meIdRealRPC, meIdResidual;
	 
	  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure(); //Anna
	  std::string folder = "DQMData/Muons/MuonSegEff/" +  folderStr->folderStructure(rpcId);

	  delete folderStr;
		
	  meIdRPC = folder +"/RPCDataOccupancyFromDT_"+ rpcsrv.name();	
	  meIdDT =folder+"/ExpectedOccupancyFromDT_"+ rpcsrv.name();

	  bxDistroId =folder+"/BXDistribution_"+ rpcsrv.name();
	  meIdRealRPC =folder+"/RealDetectedOccupancyFromDT_"+ rpcsrv.name();  

	  meIdPRO = "Profile_For_"+rpcsrv.name();
	  meIdPRO_2D = "Profile2D_For_"+rpcsrv.name();
	  meIdResidual =folder+"/RPCResidualsFromDT_"+ rpcsrv.name();
	  meIdDT_2D =folder+"/ExpectedOccupancy2DFromDT_"+ rpcsrv.name();
	  meIdRPC_2D = folder +"/RPCDataOccupancy2DFromDT_"+ rpcsrv.name();	
	  

	  if(dosD){
	    histoRPC_2D= (TH2F*)theFile->Get(meIdRPC_2D.c_str());
	    histoDT_2D= (TH2F*)theFile->Get(meIdDT_2D.c_str());
	    histoResidual= (TH1F*)theFile->Get(meIdResidual.c_str());
	  }

	  histoRPC= (TH1F*)theFile->Get(meIdRPC.c_str());
          histoDT= (TH1F*)theFile->Get(meIdDT.c_str());
          BXDistribution = (TH1F*)theFile->Get(bxDistroId.c_str());
          histoRealRPC = (TH1F*)theFile->Get(meIdRealRPC.c_str());
	  
	  histoPRO= new TH1F (meIdPRO.c_str(),meIdPRO.c_str(),int((*r)->nstrips()),0.5,int((*r)->nstrips())+0.5);
	  histoPRO_2D= new TH2F (meIdPRO_2D.c_str(),meIdPRO.c_str(),nstrips,-0.5*nstrips*stripw,0.5*nstrips*stripw,nstrips,-0.5*stripl,0.5*stripl);
	  
	  std::cout <<folder<<"/"<<rpcsrv.name()<<std::endl;

	  int NumberMasked=0;
	  int NumberWithOutPrediction=0;
	  double p = 0;
	  double o = 0;
	  float mybxhisto = 0;
	  float mybxerror = 0;
	  float ef =0;
	  float er =0;
	  float ef2D =0;
	  float er2D =0;
	  float buffef = 0;
	  float buffer = 0;
	  float sumbuffef = 0;
	  float sumbuffer = 0;
	  float averageeff = 0;
	  float averageerr = 0;

	  float doublegapeff = 0;
	  float doublegaperr = 0;

	  float bufdoublegapeff = 0;
	  float bufdoublegaperr = 0;
	  
	  int NumberStripsPointed = 0;
	  double deadStripsContribution=0;
	  
	  if(dosD && histoRPC_2D && histoDT_2D && histoResidual){
	    //std::cout<<"Leidos los histogramas 2D!"<<std::endl;
	    for(int i=1;i<=nstrips;++i){
	      for(int j=1;j<=nstrips;++j){
		if(histoDT_2D->GetBinContent(i,j) != 0){
		  ef2D = histoRPC_2D->GetBinContent(i,j)/histoDT_2D->GetBinContent(i,j);
		  er2D = sqrt(ef2D*(1-ef2D)/histoDT_2D->GetBinContent(i,j));
		}	
		histoPRO_2D->SetBinContent(i,j,ef2D*100.);
		histoPRO_2D->SetBinError(i,j,er2D*100.);
	      }//loop on the boxes
	    }
	  }else{
	    std::cout<<"Warning!!! Alguno de los  histogramas 2D no fue leido!"<<std::endl;
	  }

	  bool maskeffect[100];
	  for(int i=0;i<100;i++) maskeffect[i]=false;
	    
	  if(histoRPC && histoDT && BXDistribution && histoRealRPC){
	    int nstrips=(*r)->nstrips();
	    for(int i=1;i<=int(nstrips);++i){
	      if(histoRealRPC->GetBinContent(i)==0){
		std::cout<<"1";
		if(i==1){
		  maskeffect[1]=true;
		  maskeffect[2]=true;
		  maskeffect[3]=true;
		}else if(i==2){
		  maskeffect[1]=true;
		  maskeffect[2]=true;
		  maskeffect[3]=true;
		  maskeffect[4]=true;
		}else if(i==(*r)->nstrips()){
		  maskeffect[nstrips-2]=true;
		  maskeffect[nstrips-1]=true;
		  maskeffect[nstrips]=true;
		}else if(i==(*r)->nstrips()-1){
		  maskeffect[nstrips-3]=true;
		  maskeffect[nstrips-2]=true;
		  maskeffect[nstrips-1]=true;
		  maskeffect[nstrips]=true;
		}else{
		  maskeffect[i-2]=true;
		  maskeffect[i-1]=true;
		  maskeffect[i]=true;
		  maskeffect[i+1]=true;
		  maskeffect[i+2]=true;
		}
	      }else{
		std::cout<<"0";
	      }
	    }
	    
	    //Border Effect
	    maskeffect[nstrips]=true;
	    maskeffect[nstrips-1]=true;
	    maskeffect[nstrips-2]=true;
	    maskeffect[nstrips-3]=true;
	    maskeffect[nstrips-4]=true;
	    maskeffect[nstrips-5]=true;
	    
	    std::cout<<std::endl;
	    
	    float withouteffect=0.;
	    
	    for(int i=1;i<=int((*r)->nstrips());i++){
	      if(maskeffect[i]==false){
		withouteffect++;
		std::cout<<"0";
	      }else{
		std::cout<<"1";
	      }
	    }

	    std::cout<<std::endl;
	    
	    for(int i=1;i<=int((*r)->nstrips());++i){
	      if(histoRealRPC->GetBinContent(i)==0){
		NumberMasked++;
		deadStripsContribution=deadStripsContribution+histoDT->GetBinContent(i);
	      }
	      if(histoDT->GetBinContent(i)!=0){
		buffef = float(histoRPC->GetBinContent(i))/float(histoDT->GetBinContent(i));
		buffer = sqrt(buffef*(1.-buffef)/float(histoDT->GetBinContent(i)));
		
		std::cout<<" "<<buffef*100;
		
		sumbuffef = sumbuffef + buffef;
		sumbuffer = sumbuffer + buffer*buffer;
		NumberStripsPointed++;
		if(maskeffect[i]==false){
		  bufdoublegapeff=bufdoublegapeff+buffef;
		  bufdoublegaperr=bufdoublegaperr+buffer*buffer;
		}
	      }else{
		std::cout<<" NP";
		NumberWithOutPrediction++;
	      }
	      histoPRO->SetBinContent(i,buffef);
	      histoPRO->SetBinError(i,buffer);
	    }

	    assert(NumberWithOutPrediction+NumberStripsPointed == (*r)->nstrips());
	    
	    if(NumberStripsPointed!=0){
	      averageeff = (sumbuffef/float(NumberStripsPointed))*100.;
	      averageerr = sqrt(sumbuffer/float(NumberStripsPointed))*100.;
	      
	      EffBarrel->Fill(averageeff);
	      
	      doublegapeff=0.;
	      doublegaperr=0.;
	      if(withouteffect!=0){
		doublegapeff=(bufdoublegapeff/withouteffect)*100.;
		doublegaperr=sqrt(bufdoublegaperr/withouteffect)*100.;
	      }
	      
	      std::cout<<" Eff="<<averageeff<<" DoubleGapEff"<<doublegapeff<<std::endl;
	      
	      DoubleGapBarrel->Fill(doublegapeff);
	      
	      int Ring = rpcId.ring();
	      
	      if(sector==1||sector==2||sector==3||sector==10||sector==11||sector==12){
		if(Ring==-2){ EffDistroWm2->Fill(averageeff);       DoubleGapDistroWm2->Fill(doublegapeff);
		}else if(Ring==-1){ EffDistroWm1->Fill(averageeff); DoubleGapDistroWm1->Fill(doublegapeff);
		}else if(Ring==0) { EffDistroW0->Fill(averageeff);  DoubleGapDistroW0->Fill(doublegapeff); 
		}else if(Ring==1) { EffDistroW1->Fill(averageeff);  DoubleGapDistroW1->Fill(doublegapeff); 
		}else if(Ring==2) { EffDistroW2->Fill(averageeff);  DoubleGapDistroW2->Fill(doublegapeff); 
		}
	      }else{
		if(Ring==-2){ EffDistroWm2far->Fill(averageeff);       DoubleGapDistroWm2far->Fill(doublegapeff);
		}else if(Ring==-1){ EffDistroWm1far->Fill(averageeff); DoubleGapDistroWm1far->Fill(doublegapeff);
		}else if(Ring==0) { EffDistroW0far->Fill(averageeff);  DoubleGapDistroW0far->Fill(doublegapeff); 
		}else if(Ring==1) { EffDistroW1far->Fill(averageeff);  DoubleGapDistroW1far->Fill(doublegapeff); 
		}else if(Ring==2) { EffDistroW2far->Fill(averageeff);  DoubleGapDistroW2far->Fill(doublegapeff); 
		}
	      }
	    }else{
	      std::cout<<"This Roll Doesn't have any strip Pointed"<<std::endl;
	    }
	    
	    std::cout<<std::endl;

	    if(prodimages || makehtml){
	      command = "mkdir " + rpcsrv.name();
	      system(command.c_str());
	    }

	    histoPRO->Write();

	    if(prodimages){
	      histoPRO->GetXaxis()->SetTitle("Strip");
	      histoPRO->GetYaxis()->SetTitle("Efficiency (%)");
	      histoPRO->GetYaxis()->SetRangeUser(0.,1.);
	      histoPRO->Draw();
	      std::string labeltoSave = rpcsrv.name() + "/Profile.png";
	      Ca0->SaveAs(labeltoSave.c_str());
	      Ca0->Clear();

	      histoRPC->GetXaxis()->SetTitle("Strip");
	      histoRPC->GetYaxis()->SetTitle("Occupancy Extrapolation");
	      histoRPC->Draw();
	      labeltoSave = rpcsrv.name() + "/RPCOccupancy.png";
	      Ca0->SaveAs(labeltoSave.c_str());
	      Ca0->Clear();

	      histoRealRPC->GetXaxis()->SetTitle("Strip");
	      histoRealRPC->GetYaxis()->SetTitle("RPC Occupancy");
	      histoRealRPC->Draw();
	      labeltoSave = rpcsrv.name() + "/DQMOccupancy.png";
	      Ca0->SaveAs(labeltoSave.c_str());
	      Ca0->Clear();
	      
	      histoDT->GetXaxis()->SetTitle("Strip");
	      histoDT->GetYaxis()->SetTitle("Expected Occupancy");
	      histoDT->Draw();
	      labeltoSave = rpcsrv.name() + "/DTOccupancy.png";
	      Ca0->SaveAs(labeltoSave.c_str());
	      Ca0->Clear();
	      
	      BXDistribution->GetXaxis()->SetTitle("BX");
	      BXDistribution->Draw();
	      labeltoSave = rpcsrv.name() + "/BXDistribution.png";
	      Ca0->SaveAs(labeltoSave.c_str());
	      Ca0->Clear();
	      
	      if(dosD){
		histoRPC_2D->GetXaxis()->SetTitle("cm");
		histoRPC_2D->GetYaxis()->SetTitle("cm");
		histoRPC_2D->Draw();
		labeltoSave = rpcsrv.name() + "/RPCOccupancy_2D.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();
		
		histoDT_2D->GetXaxis()->SetTitle("cm");
		histoDT_2D->GetYaxis()->SetTitle("cm");
		histoDT_2D->Draw();
		labeltoSave = rpcsrv.name() + "/DTOccupancy_2D.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();

		histoPRO_2D->GetXaxis()->SetTitle("cm");
		histoPRO_2D->GetYaxis()->SetTitle("cm");
		histoPRO_2D->Draw();
		labeltoSave = rpcsrv.name() + "/Profile_2D.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();
		
		histoResidual->GetXaxis()->SetTitle("cm");
		histoResidual->Draw();
		labeltoSave = rpcsrv.name() + "/Residual.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();
	      }
	    }


	    int Ring = rpcId.ring();
	    int sector = rpcId.sector();
	    //Near Side

	    //std::cout<<"Before if = "<<makehtml<<std::endl;
	    if(makehtml){
	      command = "cp htmltemplates/indexLocal.html " + rpcsrv.name() + "/index.html"; system(command.c_str());
	      std::cout<<"html for "<<rpcId<<std::endl;
	      
	      std::string color = "#0000FF";
	      if(averageeff<threshold) color = "#ff4500";

	      if(sector==1||sector==2||sector==3||sector==10||sector==11||sector==12){
		if(Ring==-2){ 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexWm2near.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertWm2near.html"; system(command.c_str());
		}
		else if(Ring==-1){ 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexWm1near.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertWm1near.html"; system(command.c_str());
		}
		else if(Ring==0){ 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexW0near.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertW0near.html"; system(command.c_str());
		}
		else if(Ring==1) { 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexW1near.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertW1near.html"; system(command.c_str());
		}
		else if(Ring==2) { 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexW2near.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertW2near.html"; system(command.c_str());
		}     
	      }else{
		if(Ring==-2){ 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexWm2far.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertWm2far.html"; system(command.c_str());
		}
		else if(Ring==-1){ 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexWm1far.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertWm1far.html"; system(command.c_str());
		}
		else if(Ring==0) { 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexW0far.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertW0far.html"; system(command.c_str());
		}
		else if(Ring==1) { 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexW1far.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertW1far.html"; system(command.c_str());
		}
		else if(Ring==2) { 
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexW2far.html"; system(command.c_str());
		  command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertW2far.html"; system(command.c_str());
		}     
	      }
	    }

	    //exit(1);
	    mybxhisto = 50.+BXDistribution->GetMean()*10;
	    mybxerror = BXDistribution->GetRMS()*10;
	    
	    bxbarrel->Fill(BXDistribution->GetMean(),BXDistribution->GetRMS());
	    
	  }else{
	    std::cout<<"One of the histograms Doesn't exist for Barrel!!!"<<std::endl;
	    exit(1);
	  }
	  	  
	  p=histoDT->Integral()-deadStripsContribution;
	  o=histoRPC->Integral();
	  
	  if(p!=0){
	    ef = float(o)/float(p); 
	    er = sqrt(ef*(1.-ef)/float(p));
	  }
	  
	  ef=ef*100;
	  er=er*100;
	    
	  std::string camera = rpcsrv.name().c_str();  
	  float stripsratio = (float(NumberMasked)/float((*r)->nstrips()))*100.;
	  float nopredictionsratio = (float(NumberWithOutPrediction)/float((*r)->nstrips()))*100.;
	  
	  //Pigi Histos
	  
	  //std::cout<<"Pigi "<<camera<<" "<<rpcsrv.shortname()<<" "<<(*r)->id()<<std::endl;
	  
	  if(abs((*r)->id().ring())==2){
	    //std::cout<<rollY(rpcsrv.shortname(),rollNamesExter)<<"--"<<rpcsrv.shortname()<<std::endl;
	    if((*r)->id().ring()==2) Wheel2Summary->SetBinContent((*r)->id().sector(),rollY(rpcsrv.shortname(),rollNamesExter),averageeff);
	    else Wheelm2Summary->SetBinContent((*r)->id().sector(),rollY(rpcsrv.shortname(),rollNamesExter),averageeff);
					  
	  }else{
	    //std::cout<<rollY(rpcsrv.shortname(),rollNamesInter)<<"--"<<rpcsrv.shortname()<<std::endl; 
	    if((*r)->id().ring()==-1) Wheelm1Summary->SetBinContent((*r)->id().sector(),rollY(rpcsrv.shortname(),rollNamesInter),averageeff);
	    else if((*r)->id().ring()==0) Wheel0Summary->SetBinContent((*r)->id().sector(),rollY(rpcsrv.shortname(),rollNamesInter),averageeff);
	    else if((*r)->id().ring()==1) Wheel1Summary->SetBinContent((*r)->id().sector(),rollY(rpcsrv.shortname(),rollNamesInter),averageeff);
	  }
	  
	  
	  //std::cout<<"Filling Global with: Average Eff="<<averageeff<<" Ingegral Eff="<<ef<<" Strips Ratio"<<stripsratio<<" No Predictionratio="<<nopredictionsratio<<std::endl;
	  
	  //Near Side

	  int Ring = rpcId.ring();
	  if((sector==1||sector==2||sector==3||sector==10||sector==11||sector==12)){
	    if(Ring==-2){
	      indexWheel[0]++;
	      EffGlobWm2->SetBinContent(indexWheel[0],ef);
	      EffGlobWm2->SetBinError(indexWheel[0],er);  
	      EffGlobWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera.c_str());

	      BXGlobWm2->SetBinContent(indexWheel[0],mybxhisto);  
	      BXGlobWm2->SetBinError(indexWheel[0],mybxerror);  
	      BXGlobWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera.c_str());
	      
	      MaskedGlobWm2->SetBinContent(indexWheel[0],stripsratio);  
	      MaskedGlobWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera.c_str());

	      AverageEffWm2->SetBinContent(indexWheel[0],averageeff);
	      AverageEffWm2->SetBinError(indexWheel[0],averageerr);  
	      AverageEffWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera.c_str());

	      DoubleGapWm2->SetBinContent(indexWheel[0],doublegapeff);
	      DoubleGapWm2->SetBinError(indexWheel[0],doublegaperr);  
	      DoubleGapWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera.c_str());
	      
	      NoPredictionWm2->SetBinContent(indexWheel[0],nopredictionsratio);
              NoPredictionWm2->GetXaxis()->SetBinLabel(indexWheel[0],camera.c_str());
	    }else if(Ring==-1){
	      indexWheel[1]++;  
	      EffGlobWm1->SetBinContent(indexWheel[1],ef);  
	      EffGlobWm1->SetBinError(indexWheel[1],er);  
	      EffGlobWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera.c_str());  
	      
	      BXGlobWm1->SetBinContent(indexWheel[1],mybxhisto);  
	      BXGlobWm1->SetBinError(indexWheel[1],mybxerror);  
	      BXGlobWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera.c_str());
	      
	      MaskedGlobWm1->SetBinContent(indexWheel[1],stripsratio);  
	      MaskedGlobWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera.c_str());

	      AverageEffWm1->SetBinContent(indexWheel[1],averageeff);
	      AverageEffWm1->SetBinError(indexWheel[1],averageerr);  
	      AverageEffWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera.c_str());
	      
	      DoubleGapWm1->SetBinContent(indexWheel[1],doublegapeff);
	      DoubleGapWm1->SetBinError(indexWheel[1],doublegaperr);  
	      DoubleGapWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera.c_str());
	      
	      NoPredictionWm1->SetBinContent(indexWheel[1],nopredictionsratio);
              NoPredictionWm1->GetXaxis()->SetBinLabel(indexWheel[1],camera.c_str());

	    }else if(Ring==0){
	      indexWheel[2]++;  
	      EffGlobW0->SetBinContent(indexWheel[2],ef);  
	      EffGlobW0->SetBinError(indexWheel[2],er);  
	      EffGlobW0->GetXaxis()->SetBinLabel(indexWheel[2],camera.c_str());  
	      
	      BXGlobW0->SetBinContent(indexWheel[2],mybxhisto);  
	      BXGlobW0->SetBinError(indexWheel[2],mybxerror);  
	      BXGlobW0->GetXaxis()->SetBinLabel(indexWheel[2],camera.c_str());

	      MaskedGlobW0->SetBinContent(indexWheel[2],stripsratio);  
	      MaskedGlobW0->GetXaxis()->SetBinLabel(indexWheel[2],camera.c_str());
	      
	      AverageEffW0->SetBinContent(indexWheel[2],averageeff);
	      AverageEffW0->SetBinError(indexWheel[2],averageerr);  
	      AverageEffW0->GetXaxis()->SetBinLabel(indexWheel[2],camera.c_str());

	      DoubleGapW0->SetBinContent(indexWheel[2],doublegapeff);
	      DoubleGapW0->SetBinError(indexWheel[2],doublegaperr);  
	      DoubleGapW0->GetXaxis()->SetBinLabel(indexWheel[2],camera.c_str());
	      
	      NoPredictionW0->SetBinContent(indexWheel[2],nopredictionsratio);
              NoPredictionW0->GetXaxis()->SetBinLabel(indexWheel[2],camera.c_str());	      
	    }else if(Ring==1){
	      indexWheel[3]++;  
	      EffGlobW1->SetBinContent(indexWheel[3],ef);  
	      EffGlobW1->SetBinError(indexWheel[3],er);  
	      EffGlobW1->GetXaxis()->SetBinLabel(indexWheel[3],camera.c_str());  
	      
	      BXGlobW1->SetBinContent(indexWheel[3],mybxhisto);  
	      BXGlobW1->SetBinError(indexWheel[3],mybxerror);  
	      BXGlobW1->GetXaxis()->SetBinLabel(indexWheel[3],camera.c_str());

	      MaskedGlobW1->SetBinContent(indexWheel[3],stripsratio);  
	      MaskedGlobW1->GetXaxis()->SetBinLabel(indexWheel[3],camera.c_str());

	      AverageEffW1->SetBinContent(indexWheel[3],averageeff);
	      AverageEffW1->SetBinError(indexWheel[3],averageerr);  
	      AverageEffW1->GetXaxis()->SetBinLabel(indexWheel[3],camera.c_str());

	      DoubleGapW1->SetBinContent(indexWheel[3],doublegapeff);
	      DoubleGapW1->SetBinError(indexWheel[3],doublegaperr);  
	      DoubleGapW1->GetXaxis()->SetBinLabel(indexWheel[3],camera.c_str());
	      
	      NoPredictionW1->SetBinContent(indexWheel[3],nopredictionsratio);
              NoPredictionW1->GetXaxis()->SetBinLabel(indexWheel[3],camera.c_str());	      
	    }else if(Ring==2){
	      indexWheel[4]++;
	      EffGlobW2->SetBinContent(indexWheel[4],ef);
	      EffGlobW2->SetBinError(indexWheel[4],er);
	      EffGlobW2->GetXaxis()->SetBinLabel(indexWheel[4],camera.c_str());

	      BXGlobW2->SetBinContent(indexWheel[4],mybxhisto);  
	      BXGlobW2->SetBinError(indexWheel[4],mybxerror);  
	      BXGlobW2->GetXaxis()->SetBinLabel(indexWheel[4],camera.c_str());
	      
	      MaskedGlobW2->SetBinContent(indexWheel[4],stripsratio);  
	      MaskedGlobW2->GetXaxis()->SetBinLabel(indexWheel[4],camera.c_str());

	      AverageEffW2->SetBinContent(indexWheel[4],averageeff);
	      AverageEffW2->SetBinError(indexWheel[4],averageerr);  
	      AverageEffW2->GetXaxis()->SetBinLabel(indexWheel[4],camera.c_str());

	      DoubleGapW2->SetBinContent(indexWheel[4],doublegapeff);
	      DoubleGapW2->SetBinError(indexWheel[4],doublegaperr);  
	      DoubleGapW2->GetXaxis()->SetBinLabel(indexWheel[4],camera.c_str());
	      
	      NoPredictionW2->SetBinContent(indexWheel[4],nopredictionsratio);
              NoPredictionW2->GetXaxis()->SetBinLabel(indexWheel[4],camera.c_str());	      
	    }
	  }else{//Far Side 
	    if(Ring==-2){
	      indexWheelf[0]++;  
	      EffGlobWm2far->SetBinContent(indexWheelf[0],ef);  
	      EffGlobWm2far->SetBinError(indexWheelf[0],er);  
	      EffGlobWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera.c_str());

	      BXGlobWm2far->SetBinContent(indexWheelf[0],mybxhisto);  
	      BXGlobWm2far->SetBinError(indexWheelf[0],mybxerror);  
	      BXGlobWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera.c_str());
	      
	      MaskedGlobWm2far->SetBinContent(indexWheelf[0],stripsratio);
	      MaskedGlobWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera.c_str());
	      
	      AverageEffWm2far->SetBinContent(indexWheelf[0],averageeff);
              AverageEffWm2far->SetBinError(indexWheelf[0],averageerr);
              AverageEffWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera.c_str());

	      DoubleGapWm2far->SetBinContent(indexWheelf[0],doublegapeff);
	      DoubleGapWm2far->SetBinError(indexWheelf[0],doublegaperr);  
	      DoubleGapWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera.c_str());

              NoPredictionWm2far->SetBinContent(indexWheelf[0],nopredictionsratio);
              NoPredictionWm2far->GetXaxis()->SetBinLabel(indexWheelf[0],camera.c_str());
	      
	    }else if(Ring==-1){
	      indexWheelf[1]++;  
	      EffGlobWm1far->SetBinContent(indexWheelf[1],ef);  
	      EffGlobWm1far->SetBinError(indexWheelf[1],er);  
	      EffGlobWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera.c_str());  
	      
	      BXGlobWm1far->SetBinContent(indexWheelf[1],mybxhisto);  
	      BXGlobWm1far->SetBinError(indexWheelf[1],mybxerror);  
	      BXGlobWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera.c_str());
	      
	      MaskedGlobWm1far->SetBinContent(indexWheelf[1],stripsratio);
	      MaskedGlobWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera.c_str());

	      AverageEffWm1far->SetBinContent(indexWheelf[1],averageeff);
              AverageEffWm1far->SetBinError(indexWheelf[1],averageerr);
              AverageEffWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera.c_str());

	      DoubleGapWm1far->SetBinContent(indexWheelf[1],doublegapeff);
	      DoubleGapWm1far->SetBinError(indexWheelf[1],doublegaperr);  
	      DoubleGapWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera.c_str());

              NoPredictionWm1far->SetBinContent(indexWheelf[1],nopredictionsratio);
              NoPredictionWm1far->GetXaxis()->SetBinLabel(indexWheelf[1],camera.c_str());

	    }else  if(Ring==0){
	      indexWheelf[2]++;  
	      EffGlobW0far->SetBinContent(indexWheelf[2],ef);  
	      EffGlobW0far->SetBinError(indexWheelf[2],er);  
	      EffGlobW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera.c_str());  
	      
	      BXGlobW0far->SetBinContent(indexWheelf[2],mybxhisto);  
	      BXGlobW0far->SetBinError(indexWheelf[2],mybxerror);  
	      BXGlobW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera.c_str());

	      MaskedGlobW0far->SetBinContent(indexWheelf[2],stripsratio);
	      MaskedGlobW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera.c_str());

	      AverageEffW0far->SetBinContent(indexWheelf[2],averageeff);
              AverageEffW0far->SetBinError(indexWheelf[2],averageerr);
              AverageEffW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera.c_str());
	      
	      DoubleGapW0far->SetBinContent(indexWheelf[2],doublegapeff);
	      DoubleGapW0far->SetBinError(indexWheelf[2],doublegaperr);  
	      DoubleGapW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera.c_str());

              NoPredictionW0far->SetBinContent(indexWheelf[2],nopredictionsratio);
              NoPredictionW0far->GetXaxis()->SetBinLabel(indexWheelf[2],camera.c_str());
	    }else if(Ring==1){
	      indexWheelf[3]++;  
	      EffGlobW1far->SetBinContent(indexWheelf[3],ef);  
	      EffGlobW1far->SetBinError(indexWheelf[3],er);  
	      EffGlobW1far->GetXaxis()->SetBinLabel(indexWheelf[3],camera.c_str());  
	      
	      BXGlobW1far->SetBinContent(indexWheelf[3],mybxhisto);  
	      BXGlobW1far->SetBinError(indexWheelf[3],mybxerror);  
	      BXGlobW1far->GetXaxis()->SetBinLabel(indexWheelf[3],camera.c_str());

	      MaskedGlobW1far->SetBinContent(indexWheelf[3],stripsratio);
	      MaskedGlobW1far->GetXaxis()->SetBinLabel(indexWheelf[3],camera.c_str());
	      
	      AverageEffW1far->SetBinContent(indexWheelf[3],averageeff);
              AverageEffW1far->SetBinError(indexWheelf[3],averageerr);
              AverageEffW1far->GetXaxis()->SetBinLabel(indexWheelf[3],camera.c_str());
	      
	      DoubleGapW1far->SetBinContent(indexWheelf[3],doublegapeff);
	      DoubleGapW1far->SetBinError(indexWheelf[3],doublegaperr);  
	      DoubleGapW1far->GetXaxis()->SetBinLabel(indexWheelf[3],camera.c_str());
	      
              NoPredictionW1far->SetBinContent(indexWheelf[3],nopredictionsratio);
              NoPredictionW1far->GetXaxis()->SetBinLabel(indexWheelf[3],camera.c_str());

	    }else if(Ring==2){
	      indexWheelf[4]++;
	      EffGlobW2far->SetBinContent(indexWheelf[4],ef);
	      EffGlobW2far->SetBinError(indexWheelf[4],er);
	      EffGlobW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera.c_str());

	      BXGlobW2far->SetBinContent(indexWheelf[4],mybxhisto);  
	      BXGlobW2far->SetBinError(indexWheelf[4],mybxerror);  
	      BXGlobW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera.c_str());
	      
	      MaskedGlobW2far->SetBinContent(indexWheelf[4],stripsratio);
	      MaskedGlobW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera.c_str());

	      AverageEffW2far->SetBinContent(indexWheelf[4],averageeff);
              AverageEffW2far->SetBinError(indexWheelf[4],averageerr);
              AverageEffW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera.c_str());
	      
	      DoubleGapW2far->SetBinContent(indexWheelf[4],doublegapeff);
	      DoubleGapW2far->SetBinError(indexWheelf[4],doublegaperr);  
	      DoubleGapW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera.c_str());
	      
              NoPredictionW2far->SetBinContent(indexWheelf[4],nopredictionsratio);
              NoPredictionW2far->GetXaxis()->SetBinLabel(indexWheelf[4],camera.c_str());
	    }
	  }
	}else if(endcap&&!cosmics){//ENDCAPs
	  std::cout<<"In the EndCap"<<std::endl;

	  const TrapezoidalStripTopology* top_= dynamic_cast<const TrapezoidalStripTopology*> (&((*r)->topology()));
	  float stripl = top_->stripLength();
	  float stripw = top_->pitch();
	  
	  std::string detUnitLabel, meIdRPC, meIdRPC_2D, meIdCSC, meIdCSC_2D, meIdPRO, meIdPRO_2D, bxDistroId, meIdRealRPC,meIdResidual;
	 
	  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure(); //Anna
	  std::string folder = "DQMData/Muons/MuonSegEff/" +  folderStr->folderStructure(rpcId);

	  delete folderStr;
	
	  meIdRPC = folder +"/RPCDataOccupancyFromCSC_"+ rpcsrv.name();	
	  meIdCSC =folder+"/ExpectedOccupancyFromCSC_"+ rpcsrv.name();

	  bxDistroId =folder+"/BXDistribution_"+ rpcsrv.name();
	  meIdRealRPC =folder+"/RealDetectedOccupancyFromCSC_"+ rpcsrv.name();
	  
	  meIdPRO = "Profile_For_"+rpcsrv.name();
	  meIdPRO_2D = "Profile2D_For_"+rpcsrv.name();
	  meIdResidual =folder+"/RPCResidualsFromCSC_"+ rpcsrv.name();
	  meIdCSC_2D =folder+"/ExpectedOccupancy2DFromCSC_"+ rpcsrv.name();
	  meIdRPC_2D = folder +"/RPCDataOccupancy2DFromCSC_"+ rpcsrv.name();
	 
	  if(dosD){
	    histoRPC_2D= (TH2F*)theFile->Get(meIdRPC_2D.c_str());
	    histoCSC_2D= (TH2F*)theFile->Get(meIdCSC_2D.c_str());
	    histoResidual= (TH1F*)theFile->Get(meIdResidual.c_str());
	  }

	  
	  histoRPC= (TH1F*)theFile->Get(meIdRPC.c_str()); if(!histoRPC) std::cout<<meIdRPC<<"Doesn't exist"<<std::endl; 
	  histoCSC= (TH1F*)theFile->Get(meIdCSC.c_str());if(!histoCSC)std::cout<<meIdCSC<<"Doesn't exist"<<std::endl; 
	  BXDistribution = (TH1F*)theFile->Get(bxDistroId.c_str());if(!BXDistribution)std::cout<<BXDistribution<<"Doesn't exist"<<std::endl; 
	  histoRealRPC = (TH1F*)theFile->Get(meIdRealRPC.c_str());if(!histoRealRPC)std::cout<<meIdRealRPC<<"Doesn't exist"<<std::endl; 
	  
	  histoPRO= new TH1F (meIdPRO.c_str(),meIdPRO.c_str(),int((*r)->nstrips()),0.5,int((*r)->nstrips())+0.5);
	  histoPRO_2D= new TH2F (meIdPRO_2D.c_str(),meIdPRO.c_str(),nstrips,-0.5*nstrips*stripw,0.5*nstrips*stripw,nstrips,-0.5*stripl,0.5*stripl);
	  

	  std::cout <<folder<<"/"<<rpcsrv.name()<<std::endl;
	  
	  int NumberMasked=0;
	  int NumberWithOutPrediction=0;
	  double p = 0;
	  double o = 0;
	  float mybxhisto = 0;
	  float mybxerror = 0;
	  float ef =0;
	  float er =0;
	  float ef2D =0;
	  float er2D =0;
	  float buffef = 0;
	  float buffer = 0;
	  float sumbuffef = 0;
	  float sumbuffer = 0;
	  float averageeff = 0;
	  float averageerr = 0;
	  
	  int NumberStripsPointed = 0;
	  double deadStripsContribution = 0;
	  
	  if(dosD && histoRPC_2D && histoCSC_2D && histoResidual){
	    for(int i=1;i<=nstrips;++i){
	      for(int j=1;j<=nstrips;++j){
		if(histoCSC_2D->GetBinContent(i,j) != 0){
		  ef2D = histoRPC_2D->GetBinContent(i,j)/histoCSC_2D->GetBinContent(i,j);
		  er2D = sqrt(ef2D*(1-ef2D)/histoCSC_2D->GetBinContent(i,j));
		}	
		histoPRO_2D->SetBinContent(i,j,ef2D*100.);
		histoPRO_2D->SetBinError(i,j,er2D*100.);
	      }//loop on the boxes
	    }
	  }else{
	    std::cout<<"Warning!!! Alguno de los  histogramas 2D no fue leido!"<<std::endl;
	  }

	  
	  if(histoRPC && histoCSC && BXDistribution && histoRealRPC){
	    //std::cout<<"All Histograms Exists"<<std::endl;
	    for(int i=1;i<=int((*r)->nstrips());++i){
	      if(histoRealRPC->GetBinContent(i)==0){
		NumberMasked++;
		deadStripsContribution=deadStripsContribution+histoCSC->GetBinContent(i);
	      }
	      if(histoCSC->GetBinContent(i)!=0){
		if(histoRPC->GetBinContent(i)==0){ 
		  //std::cout<<"The RPC Data was CERO!!!!! Then Efficiency 0";
		  buffer=0.;
		  buffef=0.;
		}else{
		  buffef = double(histoRPC->GetBinContent(i))/double(histoCSC->GetBinContent(i));
		  buffer = sqrt(buffef*(1.-buffef)/double(histoCSC->GetBinContent(i)));
		}
		sumbuffef=sumbuffef+buffef;
		sumbuffer = sumbuffer + buffer*buffer;
		NumberStripsPointed++;
	      }else{
		NumberWithOutPrediction++;
	      }
	      
	      histoPRO->SetBinContent(i,buffef);
	      histoPRO->SetBinError(i,buffer);

	      //std::cout<<"\t \t Write in Histo PRO"<<histoPRO->GetBinContent(i)<<std::endl;
	      //std::cout<<"\t \t Strip="<<i<<" RealRPC="<<histoRealRPC->GetBinContent(i)<<" RPC="<<histoRPC->GetBinContent(i)<<" CSC="<<histoCSC->GetBinContent(i)<<" buffef="<<buffef<<" buffer="<<buffer<<" sumbuffef="<<sumbuffef<<" sumbuffer="<<sumbuffer<<" NumberStripsPointed="<<NumberStripsPointed<<" NumberWithOutPrediction"<<NumberWithOutPrediction<<" Number Masked="<<NumberMasked<<std::endl;
	    }
	    
	    if(NumberStripsPointed!=0){
	      averageeff = (sumbuffef/float(NumberStripsPointed))*100.;
	      averageerr = sqrt(sumbuffer/float(NumberStripsPointed))*100.;
	      EffEndCap->Fill(averageeff);
	      //Filling eff distro
	      int Disk=rpcId.station()*rpcId.region();
	      if(sector==1||sector==2||sector==6){
		if(Disk==-3){EffDistroDm3->Fill(averageeff);
		}else if(Disk==-2){EffDistroDm2->Fill(averageeff);
		}else if(Disk==-1){EffDistroDm1->Fill(averageeff);
		}else if(Disk==1){EffDistroD1->Fill(averageeff);
		}else if(Disk==2){EffDistroD2->Fill(averageeff);
		}else if(Disk==3){EffDistroD3->Fill(averageeff);
		}
	      }else{//Far Side 
		if(Disk==-3){EffDistroDm3far->Fill(averageeff);
		}else if(Disk==-2){EffDistroDm2far->Fill(averageeff);
		}else if(Disk==-1){EffDistroDm1far->Fill(averageeff);
		}else if(Disk==1){EffDistroD1far->Fill(averageeff);
		}else if(Disk==2){EffDistroD2far->Fill(averageeff);
		}else if(Disk==3){EffDistroD3far->Fill(averageeff);
		}
	      }//Finishing EndCap
	      

	      if(prodimages || makehtml){	       
		command = "mkdir " + rpcsrv.name();
		system(command.c_str());
	      }

	      histoPRO->Write();

	      if(prodimages){//ENDCAP
		histoPRO->GetXaxis()->SetTitle("Strip");
		histoPRO->GetYaxis()->SetTitle("Efficiency (%)");
		histoPRO->GetYaxis()->SetRangeUser(0.,1.);
		histoPRO->Draw();
		std::string labeltoSave = rpcsrv.name() + "/Profile.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();
	      
		histoRPC->GetXaxis()->SetTitle("Strip");
		histoRPC->GetYaxis()->SetTitle("Occupancy Extrapolation");
		histoRPC->Draw();
		labeltoSave = rpcsrv.name() + "/RPCOccupancy.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();
	    
		histoRealRPC->GetXaxis()->SetTitle("Strip");
		histoRealRPC->GetYaxis()->SetTitle("RPC Occupancy");
		histoRealRPC->Draw();
		labeltoSave = rpcsrv.name() + "/DQMOccupancy.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();
	      
		histoCSC->GetXaxis()->SetTitle("Strip");
		histoCSC->GetYaxis()->SetTitle("Expected Occupancy");
		histoCSC->Draw();
		labeltoSave = rpcsrv.name() + "/DTOccupancy.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();
	      
		BXDistribution->GetXaxis()->SetTitle("BX");
		BXDistribution->Draw();
		labeltoSave = rpcsrv.name() + "/BXDistribution.png";
		Ca0->SaveAs(labeltoSave.c_str());
		Ca0->Clear();
		
		if(dosD){
		  histoRPC_2D->GetXaxis()->SetTitle("cm");
		  histoRPC_2D->GetYaxis()->SetTitle("cm");
		  histoRPC_2D->Draw();
		  labeltoSave = rpcsrv.name() + "/RPCOccupancy_2D.png";
		  Ca0->SaveAs(labeltoSave.c_str());
		  Ca0->Clear();
		   
		  histoCSC_2D->GetXaxis()->SetTitle("cm");
		  histoCSC_2D->GetYaxis()->SetTitle("cm");
		  histoCSC_2D->Draw();
		  labeltoSave = rpcsrv.name() + "/DTOccupancy_2D.png";
		  Ca0->SaveAs(labeltoSave.c_str());
		  Ca0->Clear();
		  
		  histoPRO_2D->GetXaxis()->SetTitle("cm");
		  histoPRO_2D->GetYaxis()->SetTitle("cm");
		  histoPRO_2D->Draw();
		  labeltoSave = rpcsrv.name() + "/Profile_2D.png";
		  Ca0->SaveAs(labeltoSave.c_str());
		  Ca0->Clear();
		  
		  histoResidual->GetXaxis()->SetTitle("cm");
		  histoResidual->Draw();
		  labeltoSave = rpcsrv.name() + "/Residual.png";
		  Ca0->SaveAs(labeltoSave.c_str());
		  Ca0->Clear();
		}
	      }

	      delete histoPRO;
	      delete histoPRO_2D;

	      int sector = rpcId.sector();
	      //Near Side
	      
	      //std::cout<<"Before if = "<<makehtml<<std::endl;
	      if(makehtml){
		command = "cp htmltemplates/indexLocal.html " + rpcsrv.name() + "/index.html"; system(command.c_str());
		std::cout<<"html for "<<rpcId<<std::endl;
		
		std::string color = "#0000FF";
		if(averageeff<threshold) color = "#ff4500";
		
		int Disk=rpcId.station()*rpcId.region();
		
		if(sector==1||sector==2||sector==3){
		  if(Disk==-3){ 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexDm3near.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertDm3near.html"; system(command.c_str());
		  }
		  else if(Disk==-2){ 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/ndextemplate.html >> indexDm2near.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertDm2near.html"; system(command.c_str());
		  }
		  else if(Disk==-1){ 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexDm1near.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertDm1near.html"; system(command.c_str());
		  }
		  else if(Disk==1){ 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexD1near.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertD1near.html"; system(command.c_str());
		  }
		  else if(Disk==2) { 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexD2near.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertD2near.html"; system(command.c_str());
		  }else if(Disk==3){
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexD3near.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertD3near.html"; system(command.c_str());
		  }
		}else{
		  if(Disk==-3){ 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexDm3far.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertDm3far.html"; system(command.c_str());
		  }else if(Disk==-2){ 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexDm2far.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertDm2far.html"; system(command.c_str());
		  }else if(Disk==-1){ 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexDm1far.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertDm1far.html"; system(command.c_str());
		  }else if(Disk==1){ 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexD1far.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertD1far.html"; system(command.c_str());
		  }else if(Disk==2) { 
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexD2far.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertD2far.html"; system(command.c_str());
		  }else if(Disk==3){
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" htmltemplates/indextemplate.html >> indexD3far.html"; system(command.c_str());
		    command = "sed -e \"s|roll|" + rpcsrv.name() + "|g\" -e \"s|colore|" + color + "|g\" htmltemplates/indexline.html >> insertD3far.html"; system(command.c_str());
		  }
		}
	      }
	    }

	    
	    mybxhisto = 50.+BXDistribution->GetMean()*10;
	    mybxerror = BXDistribution->GetRMS()*10;
	    
	    bxendcap->Fill(BXDistribution->GetMean(),BXDistribution->GetRMS());

	    
	  }else{
	    std::cout<<"One of the histograms Doesn't exist!!!"<<std::endl;
	    exit(1);
	  }
	  
	  p=histoCSC->Integral()-deadStripsContribution;
	  o=histoRPC->Integral();


	  int Disk = rpcId.station()*rpcId.region();
	  
	  if(p!=0){
	    ef = float(o)/float(p); 
	    er = sqrt(ef*(1.-ef)/float(p));
	  }
	    
	  ef=ef*100;
	  er=er*100;
	    
	  std::string camera = rpcsrv.name();
	  
	  float maskedratio = (float(NumberMasked)/float((*r)->nstrips()))*100.;
	  float nopredictionsratio = (float(NumberWithOutPrediction)/float((*r)->nstrips()))*100.;
	  
	  std::cout<<"p="<<p<<" o="<<o<<std::endl;
	  std::cout<<"ef="<<ef<<" +/- er="<<er<<std::endl;
	  std::cout<<"averageeff="<<averageeff<<" +/- averageerr="<<averageerr<<std::endl;
	  std::cout<<"maskedratio="<<maskedratio<<std::endl;
	  std::cout<<"nopredictionsratio="<<nopredictionsratio<<std::endl;
	  

	  //Pigis Histogram
	  int Y=((*r)->id().ring()-2)+(*r)->id().subsector();
	  if(Disk==-3) Diskm3Summary->SetBinContent((*r)->id().sector(),Y,averageeff);
	  else if(Disk==-2) Diskm2Summary->SetBinContent((*r)->id().sector(),Y,averageeff);
	  else if(Disk==-1) Diskm1Summary->SetBinContent((*r)->id().sector(),Y,averageeff);
	  else if(Disk==1) Disk1Summary->SetBinContent((*r)->id().sector(),Y,averageeff);
	  else if(Disk==2) Disk2Summary->SetBinContent((*r)->id().sector(),Y,averageeff);
	  else if(Disk==3) Disk3Summary->SetBinContent((*r)->id().sector(),Y,averageeff);
	  
 	  //Near Side

	  if(sector==1||sector==2||sector==6){
	    if(Disk==-3){
	      indexDisk[0]++;  
	      EffGlobDm3->SetBinContent(indexDisk[0],ef);  
	      EffGlobDm3->SetBinError(indexDisk[0],er);  
	      EffGlobDm3->GetXaxis()->SetBinLabel(indexDisk[0],camera.c_str());
            
	      BXGlobDm3->SetBinContent(indexDisk[0],mybxhisto);  
	      BXGlobDm3->SetBinError(indexDisk[0],mybxerror);  
	      BXGlobDm3->GetXaxis()->SetBinLabel(indexDisk[0],camera.c_str());
	      
	      MaskedGlobDm3->SetBinContent(indexDisk[0],maskedratio);  
	      MaskedGlobDm3->GetXaxis()->SetBinLabel(indexDisk[0],camera.c_str());

	      AverageEffDm3->SetBinContent(indexDisk[0],averageeff);
	      AverageEffDm3->SetBinError(indexDisk[0],averageerr);  
	      AverageEffDm3->GetXaxis()->SetBinLabel(indexDisk[0],camera.c_str());
	      
	      NoPredictionDm3->SetBinContent(indexDisk[0],nopredictionsratio);
              NoPredictionDm3->GetXaxis()->SetBinLabel(indexDisk[0],camera.c_str());
	    }else if(Disk==-2){
	      indexDisk[1]++;  
	      EffGlobDm2->SetBinContent(indexDisk[1],ef);  
	      EffGlobDm2->SetBinError(indexDisk[1],er);  
	      EffGlobDm2->GetXaxis()->SetBinLabel(indexDisk[1],camera.c_str());

	      BXGlobDm2->SetBinContent(indexDisk[1],mybxhisto);  
	      BXGlobDm2->SetBinError(indexDisk[1],mybxerror);  
	      BXGlobDm2->GetXaxis()->SetBinLabel(indexDisk[1],camera.c_str());
	      
	      MaskedGlobDm2->SetBinContent(indexDisk[1],maskedratio);  
	      MaskedGlobDm2->GetXaxis()->SetBinLabel(indexDisk[1],camera.c_str());

	      AverageEffDm2->SetBinContent(indexDisk[1],averageeff);
	      AverageEffDm2->SetBinError(indexDisk[1],averageerr);  
	      AverageEffDm2->GetXaxis()->SetBinLabel(indexDisk[1],camera.c_str());
	      
	      NoPredictionDm2->SetBinContent(indexDisk[1],nopredictionsratio);
              NoPredictionDm2->GetXaxis()->SetBinLabel(indexDisk[1],camera.c_str());
	    }else if(Disk==-1){
	      indexDisk[2]++;  
	      EffGlobDm1->SetBinContent(indexDisk[2],ef);  
	      EffGlobDm1->SetBinError(indexDisk[2],er);  
	      EffGlobDm1->GetXaxis()->SetBinLabel(indexDisk[2],camera.c_str());  
	      
	      BXGlobDm1->SetBinContent(indexDisk[2],mybxhisto);  
	      BXGlobDm1->SetBinError(indexDisk[2],mybxerror);  
	      BXGlobDm1->GetXaxis()->SetBinLabel(indexDisk[2],camera.c_str());
	      
	      MaskedGlobDm1->SetBinContent(indexDisk[2],maskedratio);  
	      MaskedGlobDm1->GetXaxis()->SetBinLabel(indexDisk[2],camera.c_str());

	      AverageEffDm1->SetBinContent(indexDisk[2],averageeff);
	      AverageEffDm1->SetBinError(indexDisk[2],averageerr);  
	      AverageEffDm1->GetXaxis()->SetBinLabel(indexDisk[2],camera.c_str());
	      
	      NoPredictionDm1->SetBinContent(indexDisk[2],nopredictionsratio);
              NoPredictionDm1->GetXaxis()->SetBinLabel(indexDisk[2],camera.c_str());

	    }else if(Disk==1){
	      indexDisk[3]++;  
	      EffGlobD1->SetBinContent(indexDisk[3],ef);  
	      EffGlobD1->SetBinError(indexDisk[3],er);  
	      EffGlobD1->GetXaxis()->SetBinLabel(indexDisk[3],camera.c_str());  
	      
	      BXGlobD1->SetBinContent(indexDisk[3],mybxhisto);  
	      BXGlobD1->SetBinError(indexDisk[3],mybxerror);  
	      BXGlobD1->GetXaxis()->SetBinLabel(indexDisk[3],camera.c_str());

	      MaskedGlobD1->SetBinContent(indexDisk[3],maskedratio);  
	      MaskedGlobD1->GetXaxis()->SetBinLabel(indexDisk[3],camera.c_str());

	      AverageEffD1->SetBinContent(indexDisk[3],averageeff);
	      AverageEffD1->SetBinError(indexDisk[3],averageerr);  
	      AverageEffD1->GetXaxis()->SetBinLabel(indexDisk[3],camera.c_str());
	      
	      NoPredictionD1->SetBinContent(indexDisk[3],nopredictionsratio);
              NoPredictionD1->GetXaxis()->SetBinLabel(indexDisk[3],camera.c_str());	      
	    }else if(Disk==2){
	      indexDisk[4]++;
	      EffGlobD2->SetBinContent(indexDisk[4],ef);
	      EffGlobD2->SetBinError(indexDisk[4],er);
	      EffGlobD2->GetXaxis()->SetBinLabel(indexDisk[4],camera.c_str());

	      BXGlobD2->SetBinContent(indexDisk[4],mybxhisto);  
	      BXGlobD2->SetBinError(indexDisk[4],mybxerror);  
	      BXGlobD2->GetXaxis()->SetBinLabel(indexDisk[4],camera.c_str());
	      
	      MaskedGlobD2->SetBinContent(indexDisk[4],maskedratio);  
	      MaskedGlobD2->GetXaxis()->SetBinLabel(indexDisk[4],camera.c_str());

	      AverageEffD2->SetBinContent(indexDisk[4],averageeff);
	      AverageEffD2->SetBinError(indexDisk[4],averageerr);  
	      AverageEffD2->GetXaxis()->SetBinLabel(indexDisk[4],camera.c_str());
	      
	      NoPredictionD2->SetBinContent(indexDisk[4],nopredictionsratio);
              NoPredictionD2->GetXaxis()->SetBinLabel(indexDisk[4],camera.c_str());	      
	    }else if(Disk==3){
	      indexDisk[5]++;
	      EffGlobD3->SetBinContent(indexDisk[5],ef);
	      EffGlobD3->SetBinError(indexDisk[5],er);
	      EffGlobD3->GetXaxis()->SetBinLabel(indexDisk[5],camera.c_str());

	      BXGlobD3->SetBinContent(indexDisk[5],mybxhisto);  
	      BXGlobD3->SetBinError(indexDisk[5],mybxerror);  
	      BXGlobD3->GetXaxis()->SetBinLabel(indexDisk[5],camera.c_str());
	      
	      MaskedGlobD3->SetBinContent(indexDisk[5],maskedratio);  
	      MaskedGlobD3->GetXaxis()->SetBinLabel(indexDisk[5],camera.c_str());

	      AverageEffD3->SetBinContent(indexDisk[5],averageeff);
	      AverageEffD3->SetBinError(indexDisk[5],averageerr);  
	      AverageEffD3->GetXaxis()->SetBinLabel(indexDisk[5],camera.c_str());
	      
	      NoPredictionD3->SetBinContent(indexDisk[5],nopredictionsratio);
              NoPredictionD3->GetXaxis()->SetBinLabel(indexDisk[5],camera.c_str());	      
	    }
	  }else{//Far Side 
	    
	    if(Disk==-3){
	      indexDiskf[0]++;  
	      EffGlobDm3far->SetBinContent(indexDiskf[0],ef);  
	      EffGlobDm3far->SetBinError(indexDiskf[0],er);  
	      EffGlobDm3far->GetXaxis()->SetBinLabel(indexDiskf[0],camera.c_str());

	      BXGlobDm3far->SetBinContent(indexDiskf[0],mybxhisto);  
	      BXGlobDm3far->SetBinError(indexDiskf[0],mybxerror);  
	      BXGlobDm3far->GetXaxis()->SetBinLabel(indexDiskf[0],camera.c_str());
	      
	      MaskedGlobDm3far->SetBinContent(indexDiskf[0],maskedratio);
	      MaskedGlobDm3far->GetXaxis()->SetBinLabel(indexDiskf[0],camera.c_str());
	      
	      AverageEffDm3far->SetBinContent(indexDiskf[0],averageeff);
              AverageEffDm3far->SetBinError(indexDiskf[0],averageerr);
              AverageEffDm3far->GetXaxis()->SetBinLabel(indexDiskf[0],camera.c_str());

              NoPredictionDm3far->SetBinContent(indexDisk[0],nopredictionsratio);
              NoPredictionDm3far->GetXaxis()->SetBinLabel(indexDisk[0],camera.c_str());

	    }
	    else if(Disk==-2){
	      indexDiskf[1]++;  
	      EffGlobDm2far->SetBinContent(indexDiskf[1],ef);  
	      EffGlobDm2far->SetBinError(indexDiskf[1],er);  
	      EffGlobDm2far->GetXaxis()->SetBinLabel(indexDiskf[1],camera.c_str());

	      BXGlobDm2far->SetBinContent(indexDiskf[1],mybxhisto);  
	      BXGlobDm2far->SetBinError(indexDiskf[1],mybxerror);  
	      BXGlobDm2far->GetXaxis()->SetBinLabel(indexDiskf[1],camera.c_str());
	      
	      MaskedGlobDm2far->SetBinContent(indexDiskf[1],maskedratio);
	      MaskedGlobDm2far->GetXaxis()->SetBinLabel(indexDiskf[1],camera.c_str());
	      
	      AverageEffDm2far->SetBinContent(indexDiskf[1],averageeff);
              AverageEffDm2far->SetBinError(indexDiskf[1],averageerr);
              AverageEffDm2far->GetXaxis()->SetBinLabel(indexDiskf[1],camera.c_str());

              NoPredictionDm2far->SetBinContent(indexDisk[1],nopredictionsratio);
              NoPredictionDm2far->GetXaxis()->SetBinLabel(indexDisk[1],camera.c_str());

	    }else if(Disk==-1){
	      indexDiskf[2]++;  
	      EffGlobDm1far->SetBinContent(indexDiskf[2],ef);  
	      EffGlobDm1far->SetBinError(indexDiskf[2],er);  
	      EffGlobDm1far->GetXaxis()->SetBinLabel(indexDiskf[2],camera.c_str());  
	      
	      BXGlobDm1far->SetBinContent(indexDiskf[2],mybxhisto);  
	      BXGlobDm1far->SetBinError(indexDiskf[2],mybxerror);  
	      BXGlobDm1far->GetXaxis()->SetBinLabel(indexDiskf[2],camera.c_str());
	      
	      MaskedGlobDm1far->SetBinContent(indexDiskf[2],maskedratio);
	      MaskedGlobDm1far->GetXaxis()->SetBinLabel(indexDiskf[2],camera.c_str());

	      AverageEffDm1far->SetBinContent(indexDiskf[2],averageeff);
              AverageEffDm1far->SetBinError(indexDiskf[2],averageerr);
              AverageEffDm1far->GetXaxis()->SetBinLabel(indexDiskf[2],camera.c_str());

              NoPredictionDm1far->SetBinContent(indexDiskf[2],nopredictionsratio);
              NoPredictionDm1far->GetXaxis()->SetBinLabel(indexDiskf[2],camera.c_str());

	    }else if(Disk==1){
	      indexDiskf[3]++;  
	      EffGlobD1far->SetBinContent(indexDiskf[3],ef);  
	      EffGlobD1far->SetBinError(indexDiskf[3],er);  
	      EffGlobD1far->GetXaxis()->SetBinLabel(indexDiskf[3],camera.c_str());  
	      
	      BXGlobD1far->SetBinContent(indexDiskf[3],mybxhisto);  
	      BXGlobD1far->SetBinError(indexDiskf[3],mybxerror);  
	      BXGlobD1far->GetXaxis()->SetBinLabel(indexDiskf[3],camera.c_str());

	      MaskedGlobD1far->SetBinContent(indexDiskf[3],maskedratio);
	      MaskedGlobD1far->GetXaxis()->SetBinLabel(indexDiskf[3],camera.c_str());
	      
	      AverageEffD1far->SetBinContent(indexDiskf[3],averageeff);
              AverageEffD1far->SetBinError(indexDiskf[3],averageerr);
              AverageEffD1far->GetXaxis()->SetBinLabel(indexDiskf[3],camera.c_str());

              NoPredictionD1far->SetBinContent(indexDiskf[3],nopredictionsratio);
              NoPredictionD1far->GetXaxis()->SetBinLabel(indexDiskf[3],camera.c_str());

	    }else if(Disk==2){
	      indexDiskf[4]++;
	      EffGlobD2far->SetBinContent(indexDiskf[4],ef);
	      EffGlobD2far->SetBinError(indexDiskf[4],er);
	      EffGlobD2far->GetXaxis()->SetBinLabel(indexDiskf[4],camera.c_str());

	      BXGlobD2far->SetBinContent(indexDiskf[4],mybxhisto);  
	      BXGlobD2far->SetBinError(indexDiskf[4],mybxerror);  
	      BXGlobD2far->GetXaxis()->SetBinLabel(indexDiskf[4],camera.c_str());
	      
	      MaskedGlobD2far->SetBinContent(indexDiskf[4],maskedratio);
	      MaskedGlobD2far->GetXaxis()->SetBinLabel(indexDiskf[4],camera.c_str());

	      AverageEffD2far->SetBinContent(indexDiskf[4],averageeff);
              AverageEffD2far->SetBinError(indexDiskf[4],averageerr);
              AverageEffD2far->GetXaxis()->SetBinLabel(indexDiskf[4],camera.c_str());

              NoPredictionD2far->SetBinContent(indexDiskf[4],nopredictionsratio);
              NoPredictionD2far->GetXaxis()->SetBinLabel(indexDiskf[4],camera.c_str());
	    }else if(Disk==3){
	      indexDiskf[5]++;
	      EffGlobD3far->SetBinContent(indexDiskf[5],ef);
	      EffGlobD3far->SetBinError(indexDiskf[5],er);
	      EffGlobD3far->GetXaxis()->SetBinLabel(indexDiskf[5],camera.c_str());

	      BXGlobD3far->SetBinContent(indexDiskf[5],mybxhisto);  
	      BXGlobD3far->SetBinError(indexDiskf[5],mybxerror);  
	      BXGlobD3far->GetXaxis()->SetBinLabel(indexDiskf[5],camera.c_str());
	      
	      MaskedGlobD3far->SetBinContent(indexDiskf[5],maskedratio);
	      MaskedGlobD3far->GetXaxis()->SetBinLabel(indexDiskf[5],camera.c_str());

	      AverageEffD3far->SetBinContent(indexDiskf[5],averageeff);
              AverageEffD3far->SetBinError(indexDiskf[5],averageerr);
              AverageEffD3far->GetXaxis()->SetBinLabel(indexDiskf[5],camera.c_str());

              NoPredictionD3far->SetBinContent(indexDiskf[5],nopredictionsratio);
              NoPredictionD3far->GetXaxis()->SetBinLabel(indexDiskf[5],camera.c_str());
	    }
	  }//Finishing EndCap
	}
      }
    }
  }

  if(makehtml){
    command = "cat htmltemplates/indextail.html >> indexDm3near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexDm2near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexDm1near.html"; system(command.c_str());

    command = "cat htmltemplates/indextail.html >> indexD1near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexD2near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexD3near.html"; system(command.c_str());

    command = "cat htmltemplates/indextail.html >> indexDm3far.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexDm2far.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexDm1far.html"; system(command.c_str());

    command = "cat htmltemplates/indextail.html >> indexD1far.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexD2far.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexD3far.html"; system(command.c_str());

    command = "cat htmltemplates/indextail.html >> indexWm2near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexWm2far.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexWm1near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexWm1far.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexW0near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexW0far.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexW1near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexW1far.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexW2near.html"; system(command.c_str());
    command = "cat htmltemplates/indextail.html >> indexW2far.html"; system(command.c_str());
  }

  std::cout<<"Outside the loop of rolls"<<std::endl;

  Ca5->Clear();
  
  bxbarrel->Draw();
  bxbarrel->GetYaxis()->SetTitle("RMS (bx Units)");
  bxbarrel->GetXaxis()->SetTitle("Mean (bx Units)");
  Ca5->SaveAs("bxbarrel.png");
  Ca5->SaveAs("bxbarrel.root");
  
  Ca5->Clear();
  
  bxendcap->Draw();
  bxendcap->GetYaxis()->SetTitle("RMS (bx Units)");
  bxendcap->GetXaxis()->SetTitle("Mean (bx Units)");
  Ca5->SaveAs("bxendcap.png");
  Ca5->SaveAs("bxendcap.root");


  if(barrel){
    EffGlobWm2->GetXaxis()->LabelsOption("v");
    std::cout<<"Done the first Barrel"<<std::endl;
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
  
    std::cout<<"Done with Eff Glob"<<std::endl;

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

    std::cout<<"Done with Average Masked"<<std::endl;

    AverageEffWm2->GetXaxis()->LabelsOption("v");
    AverageEffWm1->GetXaxis()->LabelsOption("v");
    AverageEffW0->GetXaxis()->LabelsOption("v");
    AverageEffW1->GetXaxis()->LabelsOption("v");
    AverageEffW2->GetXaxis()->LabelsOption("v");

    AverageEffWm2far->GetXaxis()->LabelsOption("v");
    AverageEffWm1far->GetXaxis()->LabelsOption("v");
    AverageEffW0far->GetXaxis()->LabelsOption("v");
    AverageEffW1far->GetXaxis()->LabelsOption("v");
    AverageEffW2far->GetXaxis()->LabelsOption("v");
    
    NoPredictionWm2->GetXaxis()->LabelsOption("v");
    NoPredictionWm1->GetXaxis()->LabelsOption("v");
    NoPredictionW0->GetXaxis()->LabelsOption("v");
    NoPredictionW1->GetXaxis()->LabelsOption("v");
    NoPredictionW2->GetXaxis()->LabelsOption("v");
    
    NoPredictionWm2far->GetXaxis()->LabelsOption("v"); std::cout<<"Done with Wm2fa  "<<std::endl;
    NoPredictionWm1far->GetXaxis()->LabelsOption("v"); std::cout<<"Done with Wm1fa  "<<std::endl;
    NoPredictionW0far->GetXaxis()->LabelsOption("v");  std::cout<<"Done with W0far  "<<std::endl;
    NoPredictionW1far->GetXaxis()->LabelsOption("v");  std::cout<<"Done with W1far  "<<std::endl;
    NoPredictionW2far->GetXaxis()->LabelsOption("v");  std::cout<<"Done with W2far  "<<std::endl;
    
  }if(endcap){
    
    std::cout<<"Label Options"<<std::endl;
    NoPredictionDm3->GetXaxis()->LabelsOption("v");
    AverageEffDm3->GetXaxis()->LabelsOption("v");
    EffGlobDm3->GetXaxis()->LabelsOption("v");
    BXGlobDm3->GetXaxis()->LabelsOption("v");
    MaskedGlobDm3->GetXaxis()->LabelsOption("v");
    NoPredictionDm3far->GetXaxis()->LabelsOption("v");
    AverageEffDm3far->GetXaxis()->LabelsOption("v");
    EffGlobDm3far->GetXaxis()->LabelsOption("v");
    BXGlobDm3far->GetXaxis()->LabelsOption("v");
    MaskedGlobDm3far->GetXaxis()->LabelsOption("v");
    
    std::cout<<"Label Size"<<std::endl;
    NoPredictionDm3->GetXaxis()->SetLabelSize(0.03);
    AverageEffDm3->GetXaxis()->SetLabelSize(0.03);
    EffGlobDm3->GetXaxis()->SetLabelSize(0.03);
    BXGlobDm3->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobDm3->GetXaxis()->SetLabelSize(0.03);
    NoPredictionDm3far->GetXaxis()->SetLabelSize(0.03);
    AverageEffDm3far->GetXaxis()->SetLabelSize(0.03);
    EffGlobDm3far->GetXaxis()->SetLabelSize(0.03);
    BXGlobDm3far->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobDm3far->GetXaxis()->SetLabelSize(0.03);
    
    std::cout<<"Range User"<<std::endl;
    NoPredictionDm3->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffDm3->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobDm3->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobDm3->GetYaxis()->SetRangeUser(0.,100.);
    NoPredictionDm3far->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffDm3far->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobDm3far->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobDm3far->GetYaxis()->SetRangeUser(0.,100.);
    NoPredictionDm2->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobDm3far->GetYaxis()->SetRangeUser(0.,100.);

    std::cout<<"Done with Disk m 3"<<std::endl;

    NoPredictionDm2->GetXaxis()->LabelsOption("v");
    AverageEffDm2->GetXaxis()->LabelsOption("v");
    EffGlobDm2->GetXaxis()->LabelsOption("v");
    BXGlobDm2->GetXaxis()->LabelsOption("v");
    MaskedGlobDm2->GetXaxis()->LabelsOption("v");
    NoPredictionDm2far->GetXaxis()->LabelsOption("v");
    AverageEffDm2far->GetXaxis()->LabelsOption("v");
    EffGlobDm2far->GetXaxis()->LabelsOption("v");
    BXGlobDm2far->GetXaxis()->LabelsOption("v");
    MaskedGlobDm2far->GetXaxis()->LabelsOption("v");

    NoPredictionDm2->GetXaxis()->SetLabelSize(0.03);
    AverageEffDm2->GetXaxis()->SetLabelSize(0.03);
    EffGlobDm2->GetXaxis()->SetLabelSize(0.03);
    BXGlobDm2->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobDm2->GetXaxis()->SetLabelSize(0.03);
    NoPredictionDm2far->GetXaxis()->SetLabelSize(0.03);
    AverageEffDm2far->GetXaxis()->SetLabelSize(0.03);
    EffGlobDm2far->GetXaxis()->SetLabelSize(0.03);
    BXGlobDm2far->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobDm2far->GetXaxis()->SetLabelSize(0.03);

    NoPredictionDm2->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffDm2->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobDm2->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobDm2->GetYaxis()->SetRangeUser(0.,100.);
    NoPredictionDm2far->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffDm2far->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobDm2far->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobDm2far->GetYaxis()->SetRangeUser(0.,100.);

    std::cout<<"Done with Disk m 2"<<std::endl;

    NoPredictionDm1->GetXaxis()->LabelsOption("v");
    AverageEffDm1->GetXaxis()->LabelsOption("v");
    EffGlobDm1->GetXaxis()->LabelsOption("v");
    BXGlobDm1->GetXaxis()->LabelsOption("v");
    MaskedGlobDm1->GetXaxis()->LabelsOption("v");
    NoPredictionDm1far->GetXaxis()->LabelsOption("v");
    AverageEffDm1far->GetXaxis()->LabelsOption("v");
    EffGlobDm1far->GetXaxis()->LabelsOption("v");
    BXGlobDm1far->GetXaxis()->LabelsOption("v");
    MaskedGlobDm1far->GetXaxis()->LabelsOption("v");
    NoPredictionD1->GetXaxis()->LabelsOption("v");

    NoPredictionDm1->GetXaxis()->SetLabelSize(0.03);
    AverageEffDm1->GetXaxis()->SetLabelSize(0.03);
    EffGlobDm1->GetXaxis()->SetLabelSize(0.03);
    BXGlobDm1->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobDm1->GetXaxis()->SetLabelSize(0.03);
    NoPredictionDm1far->GetXaxis()->SetLabelSize(0.03);
    AverageEffDm1far->GetXaxis()->SetLabelSize(0.03);
    EffGlobDm1far->GetXaxis()->SetLabelSize(0.03);
    BXGlobDm1far->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobDm1far->GetXaxis()->SetLabelSize(0.03);

    NoPredictionDm1->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffDm1->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobDm1->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobDm1->GetYaxis()->SetRangeUser(0.,100.);
    NoPredictionDm1far->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffDm1far->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobDm1far->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobDm1far->GetYaxis()->SetRangeUser(0.,100.);

    std::cout<<"Done with Disk m1"<<std::endl;

    AverageEffD1->GetXaxis()->LabelsOption("v");
    EffGlobD1->GetXaxis()->LabelsOption("v");
    BXGlobD1->GetXaxis()->LabelsOption("v");
    MaskedGlobD1->GetXaxis()->LabelsOption("v");
    NoPredictionD1far->GetXaxis()->LabelsOption("v");
    AverageEffD1far->GetXaxis()->LabelsOption("v");
    EffGlobD1far->GetXaxis()->LabelsOption("v");
    BXGlobD1far->GetXaxis()->LabelsOption("v");
    MaskedGlobD1far->GetXaxis()->LabelsOption("v");

    NoPredictionD1->GetXaxis()->SetLabelSize(0.03);
    AverageEffD1->GetXaxis()->SetLabelSize(0.03);
    EffGlobD1->GetXaxis()->SetLabelSize(0.03);
    BXGlobD1->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobD1->GetXaxis()->SetLabelSize(0.03);
    NoPredictionD1far->GetXaxis()->SetLabelSize(0.03);
    AverageEffD1far->GetXaxis()->SetLabelSize(0.03);
    EffGlobD1far->GetXaxis()->SetLabelSize(0.03);
    BXGlobD1far->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobD1far->GetXaxis()->SetLabelSize(0.03);

    NoPredictionD1->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffD1->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobD1->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobD1->GetYaxis()->SetRangeUser(0.,100.);
    NoPredictionD1far->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffD1far->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobD1far->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobD1far->GetYaxis()->SetRangeUser(0.,100.);

    std::cout<<"Done with Disk 1"<<std::endl;

    NoPredictionD2->GetXaxis()->LabelsOption("v");
    AverageEffD2->GetXaxis()->LabelsOption("v");
    EffGlobD2->GetXaxis()->LabelsOption("v");
    BXGlobD2->GetXaxis()->LabelsOption("v");
    MaskedGlobD2->GetXaxis()->LabelsOption("v");
    NoPredictionD2far->GetXaxis()->LabelsOption("v");
    AverageEffD2far->GetXaxis()->LabelsOption("v");
    EffGlobD2far->GetXaxis()->LabelsOption("v");
    BXGlobD2far->GetXaxis()->LabelsOption("v");
    MaskedGlobD2far->GetXaxis()->LabelsOption("v");

    NoPredictionD2->GetXaxis()->SetLabelSize(0.03);
    AverageEffD2->GetXaxis()->SetLabelSize(0.03);
    EffGlobD2->GetXaxis()->SetLabelSize(0.03);
    BXGlobD2->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobD2->GetXaxis()->SetLabelSize(0.03);
    NoPredictionD2far->GetXaxis()->SetLabelSize(0.03);
    AverageEffD2far->GetXaxis()->SetLabelSize(0.03);
    EffGlobD2far->GetXaxis()->SetLabelSize(0.03);
    BXGlobD2far->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobD2far->GetXaxis()->SetLabelSize(0.03);

    NoPredictionD2->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffD2->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobD2->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobD2->GetYaxis()->SetRangeUser(0.,100.);
    NoPredictionD2far->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffD2far->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobD2far->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobD2far->GetYaxis()->SetRangeUser(0.,100.);

    std::cout<<"Done with Disk 2"<<std::endl;

    NoPredictionD3->GetXaxis()->LabelsOption("v");
    AverageEffD3->GetXaxis()->LabelsOption("v");
    EffGlobD3->GetXaxis()->LabelsOption("v");
    BXGlobD3->GetXaxis()->LabelsOption("v");
    MaskedGlobD3->GetXaxis()->LabelsOption("v");
    NoPredictionD3far->GetXaxis()->LabelsOption("v");
    AverageEffD3far->GetXaxis()->LabelsOption("v");
    EffGlobD3far->GetXaxis()->LabelsOption("v");
    BXGlobD3far->GetXaxis()->LabelsOption("v");
    MaskedGlobD3far->GetXaxis()->LabelsOption("v");
  
    NoPredictionD3->GetXaxis()->SetLabelSize(0.03);
    AverageEffD3->GetXaxis()->SetLabelSize(0.03);
    EffGlobD3->GetXaxis()->SetLabelSize(0.03);
    BXGlobD3->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobD3->GetXaxis()->SetLabelSize(0.03);
    NoPredictionD3far->GetXaxis()->SetLabelSize(0.03);
    AverageEffD3far->GetXaxis()->SetLabelSize(0.03);
    EffGlobD3far->GetXaxis()->SetLabelSize(0.03);
    BXGlobD3far->GetXaxis()->SetLabelSize(0.03);
    MaskedGlobD3far->GetXaxis()->SetLabelSize(0.03);

    NoPredictionD3->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffD3->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobD3->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobD3->GetYaxis()->SetRangeUser(0.,100.);
    NoPredictionD3far->GetYaxis()->SetRangeUser(0.,100.);
    AverageEffD3far->GetYaxis()->SetRangeUser(0.,100.);
    EffGlobD3far->GetYaxis()->SetRangeUser(0.,100.);
    MaskedGlobD3far->GetYaxis()->SetRangeUser(0.,100.);
    
  }

  std::cout<<"Efficiency Images"<<std::endl;

  pave = new TPaveText(35,119,60,102);
  TText *t1=pave->AddText("BX Distribution (Right Axis ->)");
  t1->SetTextColor(9);
  TText *t2=pave->AddText("Average Efficiency (%)");
  t2->SetTextColor(8);
  TText *t3=pave->AddText("Integral Efficiency (%)");
  //black
  TText *t4=pave->AddText("Strips without Data (ratio)"); 
  t4->SetTextColor(2);
  TText *t5=pave->AddText("Strips Never Pointed for a Segment (ratio)");
  t5->SetTextColor(5);

  pave->SetFillColor(18);
  
  t1->SetTextSize(0.019);
  t2->SetTextSize(0.019);
  t3->SetTextSize(0.019);
  t4->SetTextSize(0.019);
  t5->SetTextSize(0.019);


  Ca2->SetBottomMargin(0.4);
  
  TGaxis * bxAxis = new TGaxis(104.,0.,104.,100.,-5,5,11,"+L");
  TGaxis * bxAxisFar = new TGaxis(108.,0.,108.,100.,-5,5,11,"+L");
  TGaxis * bxAxisEndCap = new TGaxis(112.,0.,112.,100.,-5,5,11,"+L");
  
  bxAxis->SetLabelColor(9);
  bxAxis->SetName("bxAxis");
  bxAxis->SetTitle("Mean BX (bx Units)");
  bxAxis->SetTitleColor(9);
  bxAxis->CenterTitle();
 
  bxAxisFar->SetLabelColor(9);
  bxAxisFar->SetName("bxAxis");
  bxAxisFar->SetTitle("Mean BX (bx Units)");
  bxAxisFar->SetTitleColor(9);
  bxAxisFar->CenterTitle();
 
  bxAxisEndCap->SetLabelColor(9);
  bxAxisEndCap->SetName("bxAxis");
  bxAxisEndCap->SetTitle("Mean BX (bx Units)");
  bxAxisEndCap->SetTitleColor(9);
  bxAxisEndCap->CenterTitle();
  
  gStyle->SetOptStat(0);
  
  //Negative EndCap
  
  command = "mkdir Sides" ; system(command.c_str());
  command = "mkdir Distro" ; system(command.c_str());
  command = "mkdir Pigi" ; system(command.c_str());
  
  //Producing Images
 
 Ca5->Clear();
 
 Diskm3Summary->Draw(); Diskm3Summary->GetXaxis()->SetTitle("Sector");
 Diskm3Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Diskm3Summary.png"); Ca5->SaveAs("Pigi/Diskm3Summary.root");
 Ca5->Clear();
 
 Diskm2Summary->Draw(); Diskm2Summary->GetXaxis()->SetTitle("Sector");
 Diskm2Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Diskm2Summary.png"); Ca5->SaveAs("Pigi/Diskm2Summary.root");
 Ca5->Clear();
 
 Diskm1Summary->Draw(); Diskm1Summary->GetXaxis()->SetTitle("Sector");
 Diskm1Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Diskm1Summary.png"); Ca5->SaveAs("Pigi/Diskm1Summary.root");
 Ca5->Clear();
  
 Disk3Summary->Draw(); Disk3Summary->GetXaxis()->SetTitle("Sector");
 Disk3Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Disk3Summary.png"); Ca5->SaveAs("Pigi/Disk3Summary.root");
 Ca5->Clear();

 Disk2Summary->Draw(); Disk2Summary->GetXaxis()->SetTitle("Sector");
 Disk2Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Disk2Summary.png"); Ca5->SaveAs("Pigi/Disk2Summary.root");
 Ca5->Clear();

 Disk1Summary->Draw(); Disk1Summary->GetXaxis()->SetTitle("Sector");
 Disk1Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Disk1Summary.png"); Ca5->SaveAs("Pigi/Disk1Summary.root");
 Ca5->Clear();

 Wheelm2Summary->Draw(); Wheelm2Summary->GetXaxis()->SetTitle("Sector");
 Wheelm2Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Wheelm2Summary.png"); Ca5->SaveAs("Pigi/Wheelm2Summary.root");
 Ca5->Clear();
 
 Wheelm1Summary->Draw(); Wheelm1Summary->GetXaxis()->SetTitle("Sector");
 Wheelm1Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Wheelm1Summary.png"); Ca5->SaveAs("Pigi/Wheelm1Summary.root");
 Ca5->Clear();

 Wheel0Summary->Draw(); Wheel0Summary->GetXaxis()->SetTitle("Sector");
 Wheel0Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Wheel0Summary.png"); Ca5->SaveAs("Pigi/Wheel0Summary.root");
 Ca5->Clear();

 Wheel1Summary->Draw(); Wheel1Summary->GetXaxis()->SetTitle("Sector");
 Wheel1Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Wheel1Summary.png"); Ca5->SaveAs("Pigi/Wheel1Summary.root");
 Ca5->Clear();

 Wheel2Summary->Draw(); Wheel2Summary->GetXaxis()->SetTitle("Sector");
 Wheel2Summary->SetDrawOption("color");
 Ca5->SaveAs("Pigi/Wheel2Summary.png"); Ca5->SaveAs("Pigi/Wheel2Summary.root");
 Ca5->Clear();

  if(endcap){
   
   Ca2->Clear();
   
   EffGlobDm3->Draw();
   EffGlobDm3->GetYaxis()->SetTitle("%");
   
   BXGlobDm3->SetMarkerColor(9);
   BXGlobDm3->SetLineColor(9);
   BXGlobDm3->Draw("same");
   
   MaskedGlobDm3->SetMarkerColor(2);
   MaskedGlobDm3->SetLineColor(2);
   MaskedGlobDm3->Draw("same");
   
   AverageEffDm3->SetMarkerColor(8);
   AverageEffDm3->SetLineColor(8);
   AverageEffDm3->Draw("same");
   
   NoPredictionDm3->SetMarkerColor(5);
   NoPredictionDm3->SetLineColor(5);
   NoPredictionDm3->Draw("same");
   
   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm3near.png");
   Ca2->SaveAs("SegEff_Dm3near.root");
   Ca2->Clear();


   EffGlobDm2->Draw();
   EffGlobDm2->GetYaxis()->SetTitle("%");
  
   BXGlobDm2->SetMarkerColor(9);
   BXGlobDm2->SetLineColor(9);
   BXGlobDm2->Draw("same");

   MaskedGlobDm2->SetMarkerColor(2);
   MaskedGlobDm2->SetLineColor(2);
   MaskedGlobDm2->Draw("same");

   AverageEffDm2->SetMarkerColor(8);
   AverageEffDm2->SetLineColor(8);
   AverageEffDm2->Draw("same");

   NoPredictionDm2->SetMarkerColor(5);
   NoPredictionDm2->SetLineColor(5);
   NoPredictionDm2->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm2near.png");
   Ca2->SaveAs("SegEff_Dm2near.root");
   Ca2->Clear();


   EffGlobDm1->Draw();
   EffGlobDm1->GetYaxis()->SetTitle("%");
  
   BXGlobDm1->SetMarkerColor(9);
   BXGlobDm1->SetLineColor(9);
   BXGlobDm1->Draw("same");

   MaskedGlobDm1->SetMarkerColor(2);
   MaskedGlobDm1->SetLineColor(2);
   MaskedGlobDm1->Draw("same");

   AverageEffDm1->SetMarkerColor(8);
   AverageEffDm1->SetLineColor(8);
   AverageEffDm1->Draw("same");

   NoPredictionDm1->SetMarkerColor(5);
   NoPredictionDm1->SetLineColor(5);
   NoPredictionDm1->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm1near.png");
   Ca2->SaveAs("SegEff_Dm1near.root");
   Ca2->Clear();
 }
 
 //Barrel
 if(barrel){
   
   Ca2->Clear();

   EffGlobWm2->Draw();
   EffGlobWm2->GetYaxis()->SetTitle("%");
  
   BXGlobWm2->SetMarkerColor(9);
   BXGlobWm2->SetLineColor(9);
   BXGlobWm2->Draw("same");

   MaskedGlobWm2->SetMarkerColor(2);
   MaskedGlobWm2->SetLineColor(2);
   MaskedGlobWm2->Draw("same");

   AverageEffWm2->SetMarkerColor(8);
   AverageEffWm2->SetLineColor(8);
   AverageEffWm2->Draw("same");

   DoubleGapWm2->SetMarkerColor(6);
   DoubleGapWm2->SetLineColor(6);
   DoubleGapWm2->Draw("same");

   NoPredictionWm2->SetMarkerColor(5);
   NoPredictionWm2->SetLineColor(5);
   NoPredictionWm2->Draw("same");

   pave->Draw();
 
   bxAxis->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Wm2near.png");
   Ca2->SaveAs("SegEff_Wm2near.root");
   Ca2->Clear();

   EffGlobWm2far->Draw();
   EffGlobWm2far->GetYaxis()->SetTitle("%");
  
   BXGlobWm2far->SetMarkerColor(9);
   BXGlobWm2far->SetLineColor(9);
   BXGlobWm2far->Draw("same");

   MaskedGlobWm2far->SetMarkerColor(2);
   MaskedGlobWm2far->SetLineColor(2);
   MaskedGlobWm2far->Draw("same");

   AverageEffWm2far->SetMarkerColor(8);
   AverageEffWm2far->SetLineColor(8);
   AverageEffWm2far->Draw("same");

   DoubleGapWm2far->SetMarkerColor(6);
   DoubleGapWm2far->SetLineColor(6);
   DoubleGapWm2far->Draw("same");

   NoPredictionWm2far->SetMarkerColor(5);
   NoPredictionWm2far->SetLineColor(5);
   NoPredictionWm2far->Draw("same");

   pave->Draw();

   bxAxisFar->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Wm2far.png");
   Ca2->SaveAs("SegEff_Wm2far.root");
   Ca2->Clear();

   EffGlobWm1->Draw();
   EffGlobWm1->GetYaxis()->SetTitle("%");
  
   BXGlobWm1->SetMarkerColor(9);
   BXGlobWm1->SetLineColor(9);
   BXGlobWm1->Draw("same");

   MaskedGlobWm1->SetMarkerColor(2);
   MaskedGlobWm1->SetLineColor(2);
   MaskedGlobWm1->Draw("same");

   AverageEffWm1->SetMarkerColor(8);
   AverageEffWm1->SetLineColor(8);
   AverageEffWm1->Draw("same");

   DoubleGapWm1->SetMarkerColor(6);
   DoubleGapWm1->SetLineColor(6);
   DoubleGapWm1->Draw("same");
   
   NoPredictionWm1->SetMarkerColor(5);
   NoPredictionWm1->SetLineColor(5);
   NoPredictionWm1->Draw("same");
 
   pave->Draw();

   bxAxis->Draw("same");

   Ca2->SaveAs("Sides/SegEff_Wm1near.png");
   Ca2->SaveAs("SegEff_Wm1near.root");
   Ca2->Clear();

   EffGlobWm1far->Draw();
   EffGlobWm1far->GetYaxis()->SetTitle("%");
  
   BXGlobWm1far->SetMarkerColor(9);
   BXGlobWm1far->SetLineColor(9);
   BXGlobWm1far->Draw("same");

   MaskedGlobWm1far->SetMarkerColor(2);
   MaskedGlobWm1far->SetLineColor(2);
   MaskedGlobWm1far->Draw("same");

   AverageEffWm1far->SetMarkerColor(8);
   AverageEffWm1far->SetLineColor(8);
   AverageEffWm1far->Draw("same");

   DoubleGapWm1far->SetMarkerColor(6);
   DoubleGapWm1far->SetLineColor(6);
   DoubleGapWm1far->Draw("same");

   NoPredictionWm1far->SetMarkerColor(5);
   NoPredictionWm1far->SetLineColor(5);
   NoPredictionWm1far->Draw("same");
 
   pave->Draw();

   bxAxisFar->Draw("same");

   Ca2->SaveAs("Sides/SegEff_Wm1far.png");
   Ca2->SaveAs("SegEff_Wm1far.root");
   Ca2->Clear();

   EffGlobW0->Draw();
   EffGlobW0->GetYaxis()->SetTitle("%");
  
   BXGlobW0->SetMarkerColor(9);
   BXGlobW0->SetLineColor(9);
   BXGlobW0->Draw("same");

   MaskedGlobW0->SetMarkerColor(2);
   MaskedGlobW0->SetLineColor(2);
   MaskedGlobW0->Draw("same");

   AverageEffW0->SetMarkerColor(8);
   AverageEffW0->SetLineColor(8);
   AverageEffW0->Draw("same");

   DoubleGapW0->SetMarkerColor(6);
   DoubleGapW0->SetLineColor(6);
   DoubleGapW0->Draw("same");

   NoPredictionW0->SetMarkerColor(5);
   NoPredictionW0->SetLineColor(5);
   NoPredictionW0->Draw("same");
 
   pave->Draw();

   bxAxis->Draw("same");

   Ca2->SaveAs("Sides/SegEff_W0near.png");
   Ca2->SaveAs("SegEff_W0near.root");
   Ca2->Clear();

   EffGlobW0far->Draw();
   EffGlobW0far->GetYaxis()->SetTitle("%");
  
   BXGlobW0far->SetMarkerColor(9);
   BXGlobW0far->SetLineColor(9);
   BXGlobW0far->Draw("same");

   MaskedGlobW0far->SetMarkerColor(2);
   MaskedGlobW0far->SetLineColor(2);
   MaskedGlobW0far->Draw("same");

   AverageEffW0far->SetMarkerColor(8);
   AverageEffW0far->SetLineColor(8);
   AverageEffW0far->Draw("same");

   DoubleGapW0far->SetMarkerColor(6);
   DoubleGapW0far->SetLineColor(6);
   DoubleGapW0far->Draw("same");

   NoPredictionW0far->SetMarkerColor(5);
   NoPredictionW0far->SetLineColor(5);
   NoPredictionW0far->Draw("same");
 
   pave->Draw();

   bxAxisFar->Draw("same");

   Ca2->SaveAs("Sides/SegEff_W0far.png");
   Ca2->SaveAs("SegEff_W0far.root");
   Ca2->Clear();

   EffGlobW1->Draw();
   EffGlobW1->GetYaxis()->SetTitle("%");
  
   BXGlobW1->SetMarkerColor(9);
   BXGlobW1->SetLineColor(9);
   BXGlobW1->Draw("same");

   MaskedGlobW1->SetMarkerColor(2);
   MaskedGlobW1->SetLineColor(2);
   MaskedGlobW1->Draw("same");

   AverageEffW1->SetMarkerColor(8);
   AverageEffW1->SetLineColor(8);
   AverageEffW1->Draw("same");

   DoubleGapW1->SetMarkerColor(6);
   DoubleGapW1->SetLineColor(6);
   DoubleGapW1->Draw("same");

   NoPredictionW1->SetMarkerColor(5);
   NoPredictionW1->SetLineColor(5);
   NoPredictionW1->Draw("same");

   pave->Draw();

   bxAxis->Draw("same");

   Ca2->SaveAs("Sides/SegEff_W1near.png");
   Ca2->SaveAs("SegEff_W1near.root");
   Ca2->Clear();

   EffGlobW1far->Draw();
   EffGlobW1far->GetYaxis()->SetTitle("%");
  
   BXGlobW1far->SetMarkerColor(9);
   BXGlobW1far->SetLineColor(9);
   BXGlobW1far->Draw("same");

   MaskedGlobW1far->SetMarkerColor(2);
   MaskedGlobW1far->SetLineColor(2);
   MaskedGlobW1far->Draw("same");

   AverageEffW1far->SetMarkerColor(8);
   AverageEffW1far->SetLineColor(8);
   AverageEffW1far->Draw("same");

   DoubleGapW1far->SetMarkerColor(6);
   DoubleGapW1far->SetLineColor(6);
   DoubleGapW1far->Draw("same");

   NoPredictionW1far->SetMarkerColor(5);
   NoPredictionW1far->SetLineColor(5);
   NoPredictionW1far->Draw("same");

   pave->Draw();

   bxAxisFar->Draw("same");

   Ca2->SaveAs("Sides/SegEff_W1far.png");
   Ca2->SaveAs("SegEff_W1far.root");
   Ca2->Clear();

   EffGlobW2->Draw();
   EffGlobW2->GetYaxis()->SetTitle("%");
  
   BXGlobW2->SetMarkerColor(9);
   BXGlobW2->SetLineColor(9);
   BXGlobW2->Draw("same");

   MaskedGlobW2->SetMarkerColor(2);
   MaskedGlobW2->SetLineColor(2);
   MaskedGlobW2->Draw("same");

   AverageEffW2->SetMarkerColor(8);
   AverageEffW2->SetLineColor(8);
   AverageEffW2->Draw("same");

   DoubleGapW2->SetMarkerColor(6);
   DoubleGapW2->SetLineColor(6);
   DoubleGapW2->Draw("same");

   NoPredictionW2->SetMarkerColor(5);
   NoPredictionW2->SetLineColor(5);
   NoPredictionW2->Draw("same");

   pave->Draw();
  
   bxAxis->Draw("same");

   Ca2->SaveAs("Sides/SegEff_W2near.png");
   Ca2->SaveAs("SegEff_W2near.root");
   Ca2->Clear();
  
   EffGlobW2far->Draw();
   EffGlobW2far->GetYaxis()->SetTitle("%");
  
   BXGlobW2far->SetMarkerColor(9);
   BXGlobW2far->SetLineColor(9);
   BXGlobW2far->Draw("same");

   MaskedGlobW2far->SetMarkerColor(2);
   MaskedGlobW2far->SetLineColor(2);
   MaskedGlobW2far->Draw("same");

   AverageEffW2far->SetMarkerColor(8);
   AverageEffW2far->SetLineColor(8);
   AverageEffW2far->Draw("same");
   
   DoubleGapW2far->SetMarkerColor(6);
   DoubleGapW2far->SetLineColor(6);
   DoubleGapW2far->Draw("same");

   NoPredictionW2far->SetMarkerColor(5);
   NoPredictionW2far->SetLineColor(5);
   NoPredictionW2far->Draw("same");

   pave->Draw();

   bxAxisFar->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_W2far.png");
   Ca2->SaveAs("SegEff_W2far.root");
   Ca2->Clear();
 }

 //Positive EndCap

 if(endcap){
   //POSITIVE
   EffGlobD1->Draw();
   EffGlobD1->GetYaxis()->SetTitle("%");
  
   BXGlobD1->SetMarkerColor(9);
   BXGlobD1->SetLineColor(9);
   BXGlobD1->Draw("same");

   MaskedGlobD1->SetMarkerColor(2);
   MaskedGlobD1->SetLineColor(2);
   MaskedGlobD1->Draw("same");

   AverageEffD1->SetMarkerColor(8);
   AverageEffD1->SetLineColor(8);
   AverageEffD1->Draw("same");

   NoPredictionD1->SetMarkerColor(5);
   NoPredictionD1->SetLineColor(5);
   NoPredictionD1->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_D1near.png");
   Ca2->SaveAs("SegEff_D1near.root");
   Ca2->Clear();

   EffGlobD1far->Draw();
   EffGlobD1far->GetYaxis()->SetTitle("%");
  
   BXGlobD1far->SetMarkerColor(9);
   BXGlobD1far->SetLineColor(9);
   BXGlobD1far->Draw("same");

   MaskedGlobD1far->SetMarkerColor(2);
   MaskedGlobD1far->SetLineColor(2);
   MaskedGlobD1far->Draw("same");

   AverageEffD1far->SetMarkerColor(8);
   AverageEffD1far->SetLineColor(8);
   AverageEffD1far->Draw("same");

   NoPredictionD1far->SetMarkerColor(5);
   NoPredictionD1far->SetLineColor(5);
   NoPredictionD1far->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_D1far.png");
   Ca2->SaveAs("SegEff_D1far.root");
   Ca2->Clear();

   EffGlobD2->Draw();
   EffGlobD2->GetYaxis()->SetTitle("%");
  
   BXGlobD2->SetMarkerColor(9);
   BXGlobD2->SetLineColor(9);
   BXGlobD2->Draw("same");

   MaskedGlobD2->SetMarkerColor(2);
   MaskedGlobD2->SetLineColor(2);
   MaskedGlobD2->Draw("same");

   AverageEffD2->SetMarkerColor(8);
   AverageEffD2->SetLineColor(8);
   AverageEffD2->Draw("same");

   NoPredictionD2->SetMarkerColor(5);
   NoPredictionD2->SetLineColor(5);
   NoPredictionD2->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_D2near.png");
   Ca2->SaveAs("SegEff_D2near.root");
   Ca2->Clear();

   EffGlobD2far->Draw();
   EffGlobD2far->GetYaxis()->SetTitle("%");
  
   BXGlobD2far->SetMarkerColor(9);
   BXGlobD2far->SetLineColor(9);
   BXGlobD2far->Draw("same");

   MaskedGlobD2far->SetMarkerColor(2);
   MaskedGlobD2far->SetLineColor(2);
   MaskedGlobD2far->Draw("same");

   AverageEffD2far->SetMarkerColor(8);
   AverageEffD2far->SetLineColor(8);
   AverageEffD2far->Draw("same");

   NoPredictionD2far->SetMarkerColor(5);
   NoPredictionD2far->SetLineColor(5);
   NoPredictionD2far->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_D2far.png");
   Ca2->SaveAs("SegEff_D2far.root");
   Ca2->Clear();


   EffGlobD3far->Draw();
   EffGlobD3far->GetYaxis()->SetTitle("%");
  
   BXGlobD3far->SetMarkerColor(9);
   BXGlobD3far->SetLineColor(9);
   BXGlobD3far->Draw("same");

   MaskedGlobD3far->SetMarkerColor(2);
   MaskedGlobD3far->SetLineColor(2);
   MaskedGlobD3far->Draw("same");

   AverageEffD3far->SetMarkerColor(8);
   AverageEffD3far->SetLineColor(8);
   AverageEffD3far->Draw("same");

   NoPredictionD3far->SetMarkerColor(5);
   NoPredictionD3far->SetLineColor(5);
   NoPredictionD3far->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_D3far.png");
   Ca2->SaveAs("SegEff_D3far.root");
   Ca2->Clear();

   EffGlobD3->Draw();
   EffGlobD3->GetYaxis()->SetTitle("%");
  
   BXGlobD3->SetMarkerColor(9);
   BXGlobD3->SetLineColor(9);
   BXGlobD3->Draw("same");

   MaskedGlobD3->SetMarkerColor(2);
   MaskedGlobD3->SetLineColor(2);
   MaskedGlobD3->Draw("same");

   AverageEffD3->SetMarkerColor(8);
   AverageEffD3->SetLineColor(8);
   AverageEffD3->Draw("same");

   NoPredictionD3->SetMarkerColor(5);
   NoPredictionD3->SetLineColor(5);
   NoPredictionD3->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_D3near.png");
   Ca2->SaveAs("SegEff_D3near.root");
   Ca2->Clear();
   
   //NEGATIVE

   EffGlobDm1->Draw();
   EffGlobDm1->GetYaxis()->SetTitle("%");
  
   BXGlobDm1->SetMarkerColor(9);
   BXGlobDm1->SetLineColor(9);
   BXGlobDm1->Draw("same");

   MaskedGlobDm1->SetMarkerColor(2);
   MaskedGlobDm1->SetLineColor(2);
   MaskedGlobDm1->Draw("same");

   AverageEffDm1->SetMarkerColor(8);
   AverageEffDm1->SetLineColor(8);
   AverageEffDm1->Draw("same");

   NoPredictionDm1->SetMarkerColor(5);
   NoPredictionDm1->SetLineColor(5);
   NoPredictionDm1->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm1near.png");
   Ca2->SaveAs("SegEff_Dm1near.root");
   Ca2->Clear();

   EffGlobDm1far->Draw();
   EffGlobDm1far->GetYaxis()->SetTitle("%");
  
   BXGlobDm1far->SetMarkerColor(9);
   BXGlobDm1far->SetLineColor(9);
   BXGlobDm1far->Draw("same");

   MaskedGlobDm1far->SetMarkerColor(2);
   MaskedGlobDm1far->SetLineColor(2);
   MaskedGlobDm1far->Draw("same");

   AverageEffDm1far->SetMarkerColor(8);
   AverageEffDm1far->SetLineColor(8);
   AverageEffDm1far->Draw("same");

   NoPredictionDm1far->SetMarkerColor(5);
   NoPredictionDm1far->SetLineColor(5);
   NoPredictionDm1far->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm1far.png");
   Ca2->SaveAs("SegEff_Dm1far.root");
   Ca2->Clear();

   EffGlobDm2->Draw();
   EffGlobDm2->GetYaxis()->SetTitle("%");
  
   BXGlobDm2->SetMarkerColor(9);
   BXGlobDm2->SetLineColor(9);
   BXGlobDm2->Draw("same");

   MaskedGlobDm2->SetMarkerColor(2);
   MaskedGlobDm2->SetLineColor(2);
   MaskedGlobDm2->Draw("same");

   AverageEffDm2->SetMarkerColor(8);
   AverageEffDm2->SetLineColor(8);
   AverageEffDm2->Draw("same");

   NoPredictionDm2->SetMarkerColor(5);
   NoPredictionDm2->SetLineColor(5);
   NoPredictionDm2->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm2near.png");
   Ca2->SaveAs("SegEff_Dm2near.root");
   Ca2->Clear();

   EffGlobDm2far->Draw();
   EffGlobDm2far->GetYaxis()->SetTitle("%");
  
   BXGlobDm2far->SetMarkerColor(9);
   BXGlobDm2far->SetLineColor(9);
   BXGlobDm2far->Draw("same");

   MaskedGlobDm2far->SetMarkerColor(2);
   MaskedGlobDm2far->SetLineColor(2);
   MaskedGlobDm2far->Draw("same");

   AverageEffDm2far->SetMarkerColor(8);
   AverageEffDm2far->SetLineColor(8);
   AverageEffDm2far->Draw("same");

   NoPredictionDm2far->SetMarkerColor(5);
   NoPredictionDm2far->SetLineColor(5);
   NoPredictionDm2far->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm2far.png");
   Ca2->SaveAs("SegEff_Dm2far.root");
   Ca2->Clear();


   EffGlobDm3far->Draw();
   EffGlobDm3far->GetYaxis()->SetTitle("%");
  
   BXGlobDm3far->SetMarkerColor(9);
   BXGlobDm3far->SetLineColor(9);
   BXGlobDm3far->Draw("same");

   MaskedGlobDm3far->SetMarkerColor(2);
   MaskedGlobDm3far->SetLineColor(2);
   MaskedGlobDm3far->Draw("same");

   AverageEffDm3far->SetMarkerColor(8);
   AverageEffDm3far->SetLineColor(8);
   AverageEffDm3far->Draw("same");

   NoPredictionDm3far->SetMarkerColor(5);
   NoPredictionDm3far->SetLineColor(5);
   NoPredictionDm3far->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm3far.png");
   Ca2->SaveAs("SegEff_Dm3far.root");
   Ca2->Clear();

   EffGlobDm3->Draw();
   EffGlobDm3->GetYaxis()->SetTitle("%");
  
   BXGlobDm3->SetMarkerColor(9);
   BXGlobDm3->SetLineColor(9);
   BXGlobDm3->Draw("same");

   MaskedGlobDm3->SetMarkerColor(2);
   MaskedGlobDm3->SetLineColor(2);
   MaskedGlobDm3->Draw("same");

   AverageEffDm3->SetMarkerColor(8);
   AverageEffDm3->SetLineColor(8);
   AverageEffDm3->Draw("same");

   NoPredictionDm3->SetMarkerColor(5);
   NoPredictionDm3->SetLineColor(5);
   NoPredictionDm3->Draw("same");

   pave->Draw();

   bxAxisEndCap->Draw("same");
  
   Ca2->SaveAs("Sides/SegEff_Dm3near.png");
   Ca2->SaveAs("SegEff_Dm3near.root");
   Ca2->Clear();
 }

 Ca1 = new TCanvas("Ca1","Efficiency",800,600);
 
 
 if(barrel){
   EffBarrel->GetXaxis()->SetTitle("%"); EffBarrel->Draw(); Ca1->SaveAs("Distro/EffDistroBarrel.png");Ca1->SaveAs("EffDistroBarrel.root"); 
   DoubleGapBarrel->GetXaxis()->SetTitle("%"); DoubleGapBarrel->Draw(); Ca1->SaveAs("Distro/DoubleGapBarrel.png");Ca1->SaveAs("DoubleGapBarrel.root"); 
   
   EffDistroWm2->GetXaxis()->SetTitle("%"); EffDistroWm2->Draw(); Ca1->SaveAs("Distro/EffDistroWm2.png");Ca1->SaveAs("EffDistroWm2.root"); 
   EffDistroWm1->GetXaxis()->SetTitle("%"); EffDistroWm1->Draw(); Ca1->SaveAs("Distro/EffDistroWm1.png");Ca1->SaveAs("EffDistroWm1.root"); 
   EffDistroW0->GetXaxis()->SetTitle("%"); EffDistroW0->Draw(); Ca1->SaveAs("Distro/EffDistroW0.png");Ca1->SaveAs("EffDistroW0.root"); 
   EffDistroW1->GetXaxis()->SetTitle("%"); EffDistroW1->Draw(); Ca1->SaveAs("Distro/EffDistroW1.png");Ca1->SaveAs("EffDistroW1.root"); 
   EffDistroW2->GetXaxis()->SetTitle("%"); EffDistroW2->Draw(); Ca1->SaveAs("Distro/EffDistroW2.png");Ca1->SaveAs("EffDistroW2.root"); 

   EffDistroWm2far->GetXaxis()->SetTitle("%"); EffDistroWm2far->Draw(); Ca1->SaveAs("Distro/EffDistroWm2far.png");Ca1->SaveAs("EffDistroWm2far.root"); 
   EffDistroWm1far->GetXaxis()->SetTitle("%"); EffDistroWm1far->Draw(); Ca1->SaveAs("Distro/EffDistroWm1far.png");Ca1->SaveAs("EffDistroWm1far.root"); 
   EffDistroW0far->GetXaxis()->SetTitle("%"); EffDistroW0far->Draw(); Ca1->SaveAs("Distro/EffDistroW0far.png");Ca1->SaveAs("EffDistroW0far.root"); 
   EffDistroW1far->GetXaxis()->SetTitle("%"); EffDistroW1far->Draw(); Ca1->SaveAs("Distro/EffDistroW1far.png");Ca1->SaveAs("EffDistroW1far.root"); 
   EffDistroW2far->GetXaxis()->SetTitle("%"); EffDistroW2far->Draw(); Ca1->SaveAs("Distro/EffDistroW2far.png");Ca1->SaveAs("EffDistroW2far.root"); 
   
   DoubleGapDistroWm2->GetXaxis()->SetTitle("%"); DoubleGapDistroWm2->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroWm2.png");Ca1->SaveAs("DoubleGapDistroWm2.root"); 
   DoubleGapDistroWm1->GetXaxis()->SetTitle("%"); DoubleGapDistroWm1->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroWm1.png");Ca1->SaveAs("DoubleGapDistroWm1.root"); 
   DoubleGapDistroW0->GetXaxis()->SetTitle("%"); DoubleGapDistroW0->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroW0.png");Ca1->SaveAs("DoubleGapDistroW0.root"); 
   DoubleGapDistroW1->GetXaxis()->SetTitle("%"); DoubleGapDistroW1->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroW1.png");Ca1->SaveAs("DoubleGapDistroW1.root"); 
   DoubleGapDistroW2->GetXaxis()->SetTitle("%"); DoubleGapDistroW2->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroW2.png");Ca1->SaveAs("DoubleGapDistroW2.root"); 

   DoubleGapDistroWm2far->GetXaxis()->SetTitle("%"); DoubleGapDistroWm2far->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroWm2far.png");Ca1->SaveAs("DoubleGapDistroWm2far.root"); 
   DoubleGapDistroWm1far->GetXaxis()->SetTitle("%"); DoubleGapDistroWm1far->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroWm1far.png");Ca1->SaveAs("DoubleGapDistroWm1far.root"); 
   DoubleGapDistroW0far->GetXaxis()->SetTitle("%"); DoubleGapDistroW0far->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroW0far.png");Ca1->SaveAs("DoubleGapDistroW0far.root"); 
   DoubleGapDistroW1far->GetXaxis()->SetTitle("%"); DoubleGapDistroW1far->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroW1far.png");Ca1->SaveAs("DoubleGapDistroW1far.root"); 
   DoubleGapDistroW2far->GetXaxis()->SetTitle("%"); DoubleGapDistroW2far->Draw(); Ca1->SaveAs("Distro/DoubleGapDistroW2far.png");Ca1->SaveAs("DoubleGapDistroW2far.root"); 

 }

 if(endcap){
   EffEndCap->GetXaxis()->SetTitle("%"); EffEndCap->Draw(); Ca1->SaveAs("Distro/EffDistroEndCap.png");Ca1->SaveAs("EffEndCap.root"); 

   EffDistroDm3->GetXaxis()->SetTitle("%"); EffDistroDm3->Draw(); Ca1->SaveAs("Distro/EffDistroDm3.png");Ca1->SaveAs("EffDistroDm3.root");   
   EffDistroDm2->GetXaxis()->SetTitle("%"); EffDistroDm2->Draw(); Ca1->SaveAs("Distro/EffDistroDm2.png");Ca1->SaveAs("EffDistroDm2.root"); 
   EffDistroDm1->GetXaxis()->SetTitle("%"); EffDistroDm1->Draw(); Ca1->SaveAs("Distro/EffDistroDm1.png");Ca1->SaveAs("EffDistroDm1.root"); 
   EffDistroD1->GetXaxis()->SetTitle("%"); EffDistroD1->Draw(); Ca1->SaveAs("Distro/EffDistroD1.png");Ca1->SaveAs("EffDistroD1.root"); 
   EffDistroD2->GetXaxis()->SetTitle("%"); EffDistroD2->Draw(); Ca1->SaveAs("Distro/EffDistroD2.png");Ca1->SaveAs("EffDistroD2.root"); 
   EffDistroD3->GetXaxis()->SetTitle("%"); EffDistroD3->Draw(); Ca1->SaveAs("Distro/EffDistroD3.png");Ca1->SaveAs("EffDistroD3.root"); 
   
   EffDistroDm3far->GetXaxis()->SetTitle("%"); EffDistroDm3far->Draw(); Ca1->SaveAs("Distro/EffDistroDm3far.png");Ca1->SaveAs("EffDistroDm3far.root");   
   EffDistroDm2far->GetXaxis()->SetTitle("%"); EffDistroDm2far->Draw(); Ca1->SaveAs("Distro/EffDistroDm2far.png");Ca1->SaveAs("EffDistroDm2far.root"); 
   EffDistroDm1far->GetXaxis()->SetTitle("%"); EffDistroDm1far->Draw(); Ca1->SaveAs("Distro/EffDistroDm1far.png");Ca1->SaveAs("EffDistroDm1far.root"); 
   EffDistroD1far->GetXaxis()->SetTitle("%"); EffDistroD1far->Draw(); Ca1->SaveAs("Distro/EffDistroD1far.png");Ca1->SaveAs("EffDistroD1far.root"); 
   EffDistroD2far->GetXaxis()->SetTitle("%"); EffDistroD2far->Draw(); Ca1->SaveAs("Distro/EffDistroD2far.png");Ca1->SaveAs("EffDistroD2far.root"); 
   EffDistroD3far->GetXaxis()->SetTitle("%"); EffDistroD3far->Draw(); Ca1->SaveAs("Distro/EffDistroD3far.png");Ca1->SaveAs("EffDistroD3far.root"); 
 }


 
 theFileOut->cd();

 Wheelm2Summary->Write();
 Wheelm1Summary->Write();
 Wheel0Summary->Write();
 Wheel1Summary->Write();
 Wheel2Summary->Write();

 Diskm3Summary->Write();
 Diskm2Summary->Write();
 Diskm1Summary->Write();
 Disk1Summary->Write();
 Disk2Summary->Write();
 Disk3Summary->Write();

 DoubleGapDistroWm2->Write();
 DoubleGapDistroWm1->Write();
 DoubleGapDistroW0->Write();
 DoubleGapDistroW1->Write();
 DoubleGapDistroW2->Write();
 
 DoubleGapDistroWm2far->Write();
 DoubleGapDistroWm1far->Write();
 DoubleGapDistroW0far->Write();
 DoubleGapDistroW1far->Write();
 DoubleGapDistroW2far->Write();

 EffBarrel->Write();
 EffDistroWm2->Write();
 EffDistroWm1->Write();
 EffDistroW0->Write();
 EffDistroW1->Write();
 EffDistroW2->Write();

 EffDistroWm2far->Write();
 EffDistroWm1far->Write();
 EffDistroW0far->Write();
 EffDistroW1far->Write();
 EffDistroW2far->Write();

 EffEndCap->Write();

 EffDistroDm3->Write();  
 EffDistroDm2->Write();
 EffDistroDm1->Write();
 EffDistroD1->Write();
 EffDistroD2->Write();
 EffDistroD3->Write();

 EffDistroDm3far->Write();
 EffDistroDm2far->Write();
 EffDistroDm1far->Write();
 EffDistroD1far->Write();
 EffDistroD2far->Write();
 EffDistroD3far->Write();

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

 AverageEffWm2->Write();
 AverageEffWm1->Write();
 AverageEffW0->Write();
 AverageEffW1->Write();
 AverageEffW2->Write();

 AverageEffWm2far->Write();
 AverageEffWm1far->Write();
 AverageEffW0far->Write();
 AverageEffW1far->Write();
 AverageEffW2far->Write();

 NoPredictionWm2->Write();
 NoPredictionWm1->Write();
 NoPredictionW0->Write();
 NoPredictionW1->Write();
 NoPredictionW2->Write();

 NoPredictionWm2far->Write();
 NoPredictionWm1far->Write();
 NoPredictionW0far->Write();
 NoPredictionW1far->Write();
 NoPredictionW2far->Write();

 DoubleGapWm2far->Write();
 DoubleGapWm1far->Write();
 DoubleGapW0far->Write();
 DoubleGapW1far->Write();
 DoubleGapW2far->Write();

 DoubleGapWm2->Write();
 DoubleGapWm1->Write();
 DoubleGapW0->Write();
 DoubleGapW1->Write();
 DoubleGapW2->Write();

 NoPredictionDm3->Write();
 AverageEffDm3->Write();
 EffGlobDm3->Write();
 BXGlobDm3->Write();
 MaskedGlobDm3->Write();
 NoPredictionDm3far->Write();
 AverageEffDm3far->Write();
 EffGlobDm3far->Write();
 BXGlobDm3far->Write();
 MaskedGlobDm3far->Write();
 NoPredictionDm2->Write();
 AverageEffDm2->Write();
 EffGlobDm2->Write();
 BXGlobDm2->Write();
 MaskedGlobDm2->Write();
 NoPredictionDm2far->Write();
 AverageEffDm2far->Write();
 EffGlobDm2far->Write();
 BXGlobDm2far->Write();
 MaskedGlobDm2far->Write();
 NoPredictionDm1->Write();
 AverageEffDm1->Write();
 EffGlobDm1->Write();
 BXGlobDm1->Write();
 MaskedGlobDm1->Write();
 NoPredictionDm1far->Write();
 AverageEffDm1far->Write();
 EffGlobDm1far->Write();
 BXGlobDm1far->Write();
 MaskedGlobDm1far->Write();
 NoPredictionD1->Write();
 AverageEffD1->Write();
 EffGlobD1->Write();
 BXGlobD1->Write();
 MaskedGlobD1->Write();
 NoPredictionD1far->Write();
 AverageEffD1far->Write();
 EffGlobD1far->Write();
 BXGlobD1far->Write();
 MaskedGlobD1far->Write();
 NoPredictionD2->Write();
 AverageEffD2->Write();
 EffGlobD2->Write();
 BXGlobD2->Write();
 MaskedGlobD2->Write();
 NoPredictionD2far->Write();
 AverageEffD2far->Write();
 EffGlobD2far->Write();
 BXGlobD2far->Write();
 MaskedGlobD2far->Write();
 NoPredictionD3->Write();
 AverageEffD3->Write();
 EffGlobD3->Write();
 BXGlobD3->Write();
 MaskedGlobD3->Write();
 NoPredictionD3far->Write();
 AverageEffD3far->Write();
 EffGlobD3far->Write();
 BXGlobD3far->Write();
 MaskedGlobD3far->Write();

 Ca2->Close();
  
 theFileOut->Close();
 theFile->Close();

} 

void 
RPCMonitorEfficiency::endJob(){
    
}

DEFINE_FWK_MODULE(RPCMonitorEfficiency);

