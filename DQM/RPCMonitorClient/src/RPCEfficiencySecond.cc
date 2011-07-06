/***************************************
Author: 
Camilo Carrillo
Universidad de los Andes Bogota Colombia
camilo.carrilloATcern.ch
****************************************/

#include "DQM/RPCMonitorDigi/interface/utils.h"
#include "DQM/RPCMonitorClient/interface/RPCEfficiencySecond.h"
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"

// #include <memory>
// #include <string>
// #include <fstream>
// #include <iostream>

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

// #include <DataFormats/MuonDetId/interface/RPCDetId.h>

// #include "DQMServices/Core/interface/MonitorElement.h"

//#include "TH1F.h"


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

//  if(debug) std::cout<<"Booking Residuals Barrel"<<std::endl;
//  dbe->setCurrentFolder(folderPath+"ResidualsBarrel/");
 
  //Barrel
//   hGlobal2ResClu1La1 = dbe->book1D("GlobalResidualsClu1La1","RPC Residuals Layer 1 Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1La2 = dbe->book1D("GlobalResidualsClu1La2","RPC Residuals Layer 2 Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1La3 = dbe->book1D("GlobalResidualsClu1La3","RPC Residuals Layer 3 Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1La4 = dbe->book1D("GlobalResidualsClu1La4","RPC Residuals Layer 4 Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1La5 = dbe->book1D("GlobalResidualsClu1La5","RPC Residuals Layer 5 Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1La6 = dbe->book1D("GlobalResidualsClu1La6","RPC Residuals Layer 6 Cluster Size 1",101,-10.,10.);

//   hGlobal2ResClu2La1 = dbe->book1D("GlobalResidualsClu2La1","RPC Residuals Layer 1 Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2La2 = dbe->book1D("GlobalResidualsClu2La2","RPC Residuals Layer 2 Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2La3 = dbe->book1D("GlobalResidualsClu2La3","RPC Residuals Layer 3 Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2La4 = dbe->book1D("GlobalResidualsClu2La4","RPC Residuals Layer 4 Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2La5 = dbe->book1D("GlobalResidualsClu2La5","RPC Residuals Layer 5 Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2La6 = dbe->book1D("GlobalResidualsClu2La6","RPC Residuals Layer 6 Cluster Size 2",101,-10.,10.);

//   hGlobal2ResClu3La1 = dbe->book1D("GlobalResidualsClu3La1","RPC Residuals Layer 1 Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3La2 = dbe->book1D("GlobalResidualsClu3La2","RPC Residuals Layer 2 Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3La3 = dbe->book1D("GlobalResidualsClu3La3","RPC Residuals Layer 3 Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3La4 = dbe->book1D("GlobalResidualsClu3La4","RPC Residuals Layer 4 Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3La5 = dbe->book1D("GlobalResidualsClu3La5","RPC Residuals Layer 5 Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3La6 = dbe->book1D("GlobalResidualsClu3La6","RPC Residuals Layer 6 Cluster Size 3",101,-10.,10.);

//  if(debug) std::cout<<"Booking Residuals EndCaps"<<std::endl;
//  dbe->setCurrentFolder(folderPath+"ResidualsEndCaps/");

  //Endcap  
//   hGlobal2ResClu1R3C = dbe->book1D("GlobalResidualsClu1R3C","RPC Residuals Ring 3 Roll C Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1R3B = dbe->book1D("GlobalResidualsClu1R3B","RPC Residuals Ring 3 Roll B Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1R3A = dbe->book1D("GlobalResidualsClu1R3A","RPC Residuals Ring 3 Roll A Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1R2C = dbe->book1D("GlobalResidualsClu1R2C","RPC Residuals Ring 2 Roll C Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1R2B = dbe->book1D("GlobalResidualsClu1R2B","RPC Residuals Ring 2 Roll B Cluster Size 1",101,-10.,10.);
//   hGlobal2ResClu1R2A = dbe->book1D("GlobalResidualsClu1R2A","RPC Residuals Ring 2 Roll A Cluster Size 1",101,-10.,10.);
  
//   hGlobal2ResClu2R3C = dbe->book1D("GlobalResidualsClu2R3C","RPC Residuals Ring 3 Roll C Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2R3B = dbe->book1D("GlobalResidualsClu2R3B","RPC Residuals Ring 3 Roll B Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2R3A = dbe->book1D("GlobalResidualsClu2R3A","RPC Residuals Ring 3 Roll A Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2R2C = dbe->book1D("GlobalResidualsClu2R2C","RPC Residuals Ring 2 Roll C Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2R2B = dbe->book1D("GlobalResidualsClu2R2B","RPC Residuals Ring 2 Roll B Cluster Size 2",101,-10.,10.);
//   hGlobal2ResClu2R2A = dbe->book1D("GlobalResidualsClu2R2A","RPC Residuals Ring 2 Roll A Cluster Size 2",101,-10.,10.);
  
//   hGlobal2ResClu3R3C = dbe->book1D("GlobalResidualsClu3R3C","RPC Residuals Ring 3 Roll C Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3R3B = dbe->book1D("GlobalResidualsClu3R3B","RPC Residuals Ring 3 Roll B Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3R3A = dbe->book1D("GlobalResidualsClu3R3A","RPC Residuals Ring 3 Roll A Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3R2C = dbe->book1D("GlobalResidualsClu3R2C","RPC Residuals Ring 2 Roll C Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3R2B = dbe->book1D("GlobalResidualsClu3R2B","RPC Residuals Ring 2 Roll B Cluster Size 3",101,-10.,10.);
//   hGlobal2ResClu3R2A = dbe->book1D("GlobalResidualsClu3R2A","RPC Residuals Ring 2 Roll A Cluster Size 3",101,-10.,10.);

//  if(debug) std::cout<<"Booking EffDistros"<<std::endl;
//  dbe->setCurrentFolder(folderPath);

//   if(debug) std::cout<<"Booking statistics2"<<std::endl;
//   dbe->setCurrentFolder(folderPath);
//   statistics2 = dbe->book1D("AllStatistics","Analyzed Events DT and CSC Segments",33,0.5,33.5);
  
  //Barrel 
  
  dbe->setCurrentFolder(folderPath+"Wheel_-2");
  EffDistroWm2=dbe->book1D("EffDistroWheel_-2","Efficiency Distribution for Wheel -2 ",20,0.5,100.5);
  //EffDistroWm2far=dbe->book1D("EffDistroWheel_-2far","Efficiency Distribution for Far Side Wheel -2 ",20,0.5,100.5);
  EffGlobWm2=dbe->book1D("GlobEfficiencyWheel_-2","Efficiency Wheel -2 ",206,0.5,206.5);
  //EffGlobWm2far=dbe->book1D("GlobEfficiencyWheel_-2far","Efficiency Far Side Wheel -2",105,0.5,105.5);
  //BXGlobWm2= dbe->book1D("GlobBXWheel_-2near","BX Near Side Wheel -2",101,0.5,101.5);
  //BXGlobWm2far= dbe->book1D("GlobBXWheel_-2far","BX Far Side Wheel -2",105,0.5,105.5);
  //MaskedGlobWm2= dbe->book1D("GlobMaskedWheel_-2near","Masked Near Side Wheel -2",101,0.5,101.5);
  //MaskedGlobWm2far= dbe->book1D("GlobMaskedWheel_-2far","Masked Far Side Wheel -2",105,0.5,105.5);
  //AverageEffWm2=dbe->book1D("AverageEfficiencyWheel_-2near","Average Efficiency Near Side Wheel -2 ",101,0.5,101.5);
  //AverageEffWm2far =dbe->book1D("AverageEfficiencyWheel_-2far","Average Efficiency Far Side Wheel -2 ",105,0.5,105.5);
  //NoPredictionWm2=dbe->book1D("NoPredictionWheel_-2near","No Predictions Near Side Wheel -2 ",101,0.5,101.5);
  //NoPredictionWm2far=dbe->book1D("NoPredictionWheel_-2far","No Predictions Efficiency Far Side Wheel -2 ",105,0.5,105.5);
  
  dbe->setCurrentFolder(folderPath+"Wheel_-1");
  EffDistroWm1=dbe->book1D("EffDistroWheel_-1","Efficiency Distribution for Wheel -1 ",20,0.5,100.5);
  //EffDistroWm1far=dbe->book1D("EffDistroWheel_-1far","Efficiency Distribution for Far Side Wheel -1 ",20,0.5,100.5);
  EffGlobWm1= dbe->book1D("GlobEfficiencyWheel_-1","Efficiency Wheel -1",206,0.5,206.5);
  //EffGlobWm1far=dbe->book1D("GlobEfficiencyWheel_-1far","Efficiency Far Side Wheel -1",105,0.5,105.5);
  //BXGlobWm1= dbe->book1D("GlobBXWheel_-1near","BX Near Side Wheel -1",101,0.5,101.5);
  //BXGlobWm1far= dbe->book1D("GlobBXWheel_-1far","BX Far Side Wheel -1",105,0.5,105.5);
  //MaskedGlobWm1= dbe->book1D("GlobMaskedWheel_-1near","Masked Near Side Wheel -1",101,0.5,101.5);
  //MaskedGlobWm1far= dbe->book1D("GlobMaskedWheel_-1far","Masked Far Side Wheel -1",105,0.5,105.5);
  //AverageEffWm1=dbe->book1D("AverageEfficiencyWheel_-1near","Average Efficiency Near Side Wheel -1 ",101,0.5,101.5);
  //AverageEffWm1far=dbe->book1D("AverageEfficiencyWheel_-1far","Average Efficiency Far Side Wheel -1 ",105,0.5,105.5);
  //NoPredictionWm1=dbe->book1D("NoPredictionWheel_-1near","No Predictions Near Side Wheel -1 ",101,0.5,101.5);
  //NoPredictionWm1far=dbe->book1D("NoPredictionWheel_-1far","No Predictions Efficiency Far Side Wheel -1 ",105,0.5,105.5);

  dbe->setCurrentFolder(folderPath+"Wheel_0");
  EffDistroW0=dbe->book1D("EffDistroWheel_0","Efficiency Distribution for Wheel 0 ",20,0.5,100.5);
  //EffDistroW0far=dbe->book1D("EffDistroWheel_0far","Efficiency Distribution for Far Side Wheel 0 ",20,0.5,100.5);
  EffGlobW0 = dbe->book1D("GlobEfficiencyWheel_0","Efficiency Wheel 0",206,0.5,206.5);
  //EffGlobW0far =dbe->book1D("GlobEfficiencyWheel_0far","Efficiency Far Side Wheel 0",105,0.5,105.5);
  //BXGlobW0 = dbe->book1D("GlobBXWheel_0near","BX Near Side Wheel 0",101,0.5,101.5);
  //BXGlobW0far = dbe->book1D("GlobBXWheel_0far","BX Far Side Wheel 0",105,0.5,105.5);
  //MaskedGlobW0 = dbe->book1D("GlobMaskedWheel_0near","Masked Near Side Wheel 0",101,0.5,101.5);
  //MaskedGlobW0far = dbe->book1D("GlobMaskedWheel_0far","Masked Far Side Wheel 0",105,0.5,105.5);
  //AverageEffW0=dbe->book1D("AverageEfficiencyWheel_0near","Average Efficiency Near Side Wheel 0 ",101,0.5,101.5);
  //AverageEffW0far=dbe->book1D("AverageEfficiencyWheel_0far","Average Efficiency Far Side Wheel 0 ",105,0.5,105.5);
  //NoPredictionW0=dbe->book1D("NoPredictionWheel_0near","No Predictions Near Side Wheel 0 ",101,0.5,101.5);
  //NoPredictionW0far=dbe->book1D("NoPredictionWheel_0far","No Predictions Efficiency Far Side Wheel 0 ",105,0.5,105.5);

  dbe->setCurrentFolder(folderPath+"Wheel_1");
  EffDistroW1=dbe->book1D("EffDistroWheel_1","Efficiency Distribution for Wheel 1 ",20,0.5,100.5);
  //EffDistroW1far=dbe->book1D("EffDistroWheel_1far","Efficiency Distribution for Far Side Wheel 1 ",20,0.5,100.5);
  EffGlobW1 = dbe->book1D("GlobEfficiencyWheel_1","Efficiency Wheel 1",206,0.5,206.5);
  //EffGlobW1far =dbe->book1D("GlobEfficiencyWheel_1far","Efficiency Far Side Wheel 1",105,0.5,105.5);  
  //BXGlobW1 = dbe->book1D("GlobBXWheel_1near","BX Near Side Wheel 1",101,0.5,101.5);
  //BXGlobW1far = dbe->book1D("GlobBXWheel_1far","BX Far Side Wheel 1",105,0.5,105.5);
  //MaskedGlobW1 = dbe->book1D("GlobMaskedWheel_1near","Masked Near Side Wheel 1",101,0.5,101.5);
  //MaskedGlobW1far = dbe->book1D("GlobMaskedWheel_1far","Masked Far Side Wheel 1",105,0.5,105.5);
  //AverageEffW1=dbe->book1D("AverageEfficiencyWheel_1near","Average Efficiency Near Side Wheel 1 ",101,0.5,101.5);
  //AverageEffW1far=dbe->book1D("AverageEfficiencyWheel_1far","Average Efficiency Far Side Wheel 1 ",105,0.5,105.5);
  //NoPredictionW1=dbe->book1D("NoPredictionWheel_1near","No Predictions Near Side Wheel 1 ",101,0.5,101.5);
  //NoPredictionW1far=dbe->book1D("NoPredictionWheel_1far","No Predictions Efficiency Far Side Wheel 1 ",105,0.5,105.5);

  dbe->setCurrentFolder(folderPath+"Wheel_2");
  EffDistroW2=dbe->book1D("EffDistroWheel_2","Efficiency Distribution for Wheel 2 ",20,0.5,100.5);
  //EffDistroW2far=dbe->book1D("EffDistroWheel_2far","Efficiency Distribution for Far Side Wheel 2 ",20,0.5,100.5);
  EffGlobW2 = dbe->book1D("GlobEfficiencyWheel_2","Efficiency Wheel 2",206,0.5,206.5);
  //EffGlobW2far =dbe->book1D("GlobEfficiencyWheel_2far","Efficiency Far Side Wheel 2",105,0.5,105.5);
  //BXGlobW2 = dbe->book1D("GlobBXWheel_2near","BX Near Side Wheel 2",101,0.5,101.5);
  //BXGlobW2far = dbe->book1D("GlobBXWheel_2far","BX Far Side Wheel 2",105,0.5,105.5);
  //MaskedGlobW2 = dbe->book1D("GlobMaskedWheel_2near","Masked Near Side Wheel 2",101,0.5,101.5);
  //MaskedGlobW2far = dbe->book1D("GlobMaskedWheel_2far","Masked Far Side Wheel 2",105,0.5,105.5);
  //AverageEffW2=dbe->book1D("AverageEfficiencyWheel_2near","Average Efficiency Near Side Wheel 2 ",101,0.5,101.5);
  //AverageEffW2far=dbe->book1D("AverageEfficiencyWheel_2far","Average Efficiency Far Side Wheel 2 ",105,0.5,105.5);
  //NoPredictionW2=dbe->book1D("NoPredictionWheel_2near","No Predictions Near Side Wheel 2 ",101,0.5,101.5);
  //NoPredictionW2far=dbe->book1D("NoPredictionWheel_2far","No Predictions Efficiency Far Side Wheel 2 ",105,0.5,105.5);

  //EndCap

  dbe->setCurrentFolder(folderPath+"Disk_3");
  EffDistroD3=dbe->book1D("EffDistroDisk_3","Efficiency Distribution Disk 3 ",20,0.5,100.5);
  //EffDistroD3far=dbe->book1D("EffDistroDisk_3far","Efficiency Distribution Far Side Disk 3 ",20,0.5,100.5);
  EffGlobD3 = dbe->book1D("GlobEfficiencyDisk_3","Efficiency Disk 3",218,0.5,218.5);
  //EffGlobD3far =dbe->book1D("GlobEfficiencyDisk_3far","Efficiency Far Side Disk 3",109,0.5,109.5);
  //BXGlobD3 = dbe->book1D("GlobBXDisk_3near","BX Near Side Disk 3",109,0.5,109.5);
  //BXGlobD3far = dbe->book1D("GlobBXDisk_3far","BX Far Side Disk 3",109,0.5,109.5);
  //MaskedGlobD3 = dbe->book1D("GlobMaskedDisk_3near","Masked Near Side Disk 3",109,0.5,109.5);
  //MaskedGlobD3far = dbe->book1D("GlobMaskedDisk_3far","Masked Far Side Disk 3",109,0.5,109.5);
  //AverageEffD3=dbe->book1D("AverageEfficiencyDisk_3near","Average Efficiency Near Side Disk 3 ",109,0.5,109.5);
  //AverageEffD3far=dbe->book1D("AverageEfficiencyDisk_3far","Average Efficiency Far Side Disk 3 ",109,0.5,109.5);
  //NoPredictionD3=dbe->book1D("NoPredictionDisk_3near","No Predictions Near Side Disk 3 ",109,0.5,109.5);
  //NoPredictionD3far=dbe->book1D("NoPredictionDisk_3far","No Predictions Efficiency Far Side Disk 3 ",109,0.5,109.5);

  dbe->setCurrentFolder(folderPath+"Disk_2");
  EffDistroD2=dbe->book1D("EffDistroDisk_2","Efficiency Distribution Disk 2 ",20,0.5,100.5);
  //EffDistroD2far=dbe->book1D("EffDistroDisk_2far","Efficiency Distribution Far Side Disk 2 ",20,0.5,100.5);
  EffGlobD2 = dbe->book1D("GlobEfficiencyDisk_2","Efficiency Disk 2",218,0.5,218.5);
  //EffGlobD2far =dbe->book1D("GlobEfficiencyDisk_2far","Efficiency Far Side Disk 2",109,0.5,109.5);
  //BXGlobD2 = dbe->book1D("GlobBXDisk_2near","BX Near Side Disk 2",109,0.5,109.5);
  //BXGlobD2far = dbe->book1D("GlobBXDisk_2far","BX Far Side Disk 2",109,0.5,109.5);
  //MaskedGlobD2 = dbe->book1D("GlobMaskedDisk_2near","Masked Near Side Disk 2",109,0.5,109.5);
  //MaskedGlobD2far = dbe->book1D("GlobMaskedDisk_2far","Masked Far Side Disk 2",109,0.5,109.5);
  //AverageEffD2=dbe->book1D("AverageEfficiencyDisk_2near","Average Efficiency Near Side Disk 2 ",109,0.5,109.5);
  //AverageEffD2far=dbe->book1D("AverageEfficiencyDisk_2far","Average Efficiency Far Side Disk 2 ",109,0.5,109.5);
  //NoPredictionD2=dbe->book1D("NoPredictionDisk_2near","No Predictions Near Side Disk 2 ",109,0.5,109.5);
  //NoPredictionD2far=dbe->book1D("NoPredictionDisk_2far","No Predictions Efficiency Far Side Disk 2 ",109,0.5,109.5);

  dbe->setCurrentFolder(folderPath+"Disk_1");
  EffDistroD1=dbe->book1D("EffDistroDisk_1","Efficiency Distribution Disk 1 ",20,0.5,100.5);
  //EffDistroD1far=dbe->book1D("EffDistroDisk_1far","Efficiency Distribution Far Side Disk 1 ",20,0.5,100.5);
  EffGlobD1 = dbe->book1D("GlobEfficiencyDisk_1","Efficiency Disk 1",218,0.5,218.5);
  //EffGlobD1far =dbe->book1D("GlobEfficiencyDisk_1far","Efficiency Far Side Disk 1",109,0.5,109.5);
  //BXGlobD1 = dbe->book1D("GlobBXDisk_1near","BX Near Side Disk 1",109,0.5,109.5);
  //BXGlobD1far = dbe->book1D("GlobBXDisk_1far","BX Far Side Disk 1",109,0.5,109.5);
  //MaskedGlobD1 = dbe->book1D("GlobMaskedDisk_1near","Masked Near Side Disk 1",109,0.5,109.5);
  //MaskedGlobD1far = dbe->book1D("GlobMaskedDisk_1far","Masked Far Side Disk 1",109,0.5,109.5);
  //AverageEffD1=dbe->book1D("AverageEfficiencyDisk_1near","Average Efficiency Near Side Disk 1 ",109,0.5,109.5);
  //AverageEffD1far=dbe->book1D("AverageEfficiencyDisk_1far","Average Efficiency Far Side Disk 1 ",109,0.5,109.5);
  //NoPredictionD1=dbe->book1D("NoPredictionDisk_1near","No Predictions Near Side Disk 1 ",109,0.5,109.5);
  //NoPredictionD1far=dbe->book1D("NoPredictionDisk_1far","No Predictions Efficiency Far Side Disk 1 ",109,0.5,109.5);

  dbe->setCurrentFolder(folderPath+"Disk_-1");
  EffDistroDm1=dbe->book1D("EffDistroDisk_m1","Efficiency Distribution Disk - 1 ",20,0.5,100.5);
  //EffDistroDm1far=dbe->book1D("EffDistroDisk_m1far","Efficiency Distribution Far Side Disk - 1 ",20,0.5,100.5);
  EffGlobDm1 = dbe->book1D("GlobEfficiencyDisk_m1","Efficiency Disk -1",218,0.5,218.5);
  //EffGlobDm1far =dbe->book1D("GlobEfficiencyDisk_m1far","Efficiency Far Side Disk -1",109,0.5,109.5);
  //BXGlobDm1 = dbe->book1D("GlobBXDisk_m1near","BX Near Side Disk -1",109,0.5,109.5);
  //BXGlobDm1far = dbe->book1D("GlobBXDisk_m1far","BX Far Side Disk -1",109,0.5,109.5);
  //MaskedGlobDm1 = dbe->book1D("GlobMaskedDisk_m1near","Masked Near Side Disk -1",109,0.5,109.5);
  //MaskedGlobDm1far = dbe->book1D("GlobMaskedDisk_m1far","Masked Far Side Disk -1",109,0.5,109.5);
  //AverageEffDm1=dbe->book1D("AverageEfficiencyDisk_m1near","Average Efficiency Near Side Disk -1 ",109,0.5,109.5);
  //AverageEffDm1far=dbe->book1D("AverageEfficiencyDisk_m1far","Average Efficiency Far Side Disk -1 ",109,0.5,109.5);
  //NoPredictionDm1=dbe->book1D("NoPredictionDisk_m1near","No Predictions Near Side Disk -1 ",109,0.5,109.5);
  //NoPredictionDm1far=dbe->book1D("NoPredictionDisk_m1far","No Predictions Efficiency Far Side Disk -1 ",109,0.5,109.5);

  dbe->setCurrentFolder(folderPath+"Disk_-2");
  EffDistroDm2=dbe->book1D("EffDistroDisk_m2","Efficiency Distribution Disk - 2 ",20,0.5,100.5);
  //EffDistroDm2far=dbe->book1D("EffDistroDisk_m2far","Efficiency Distribution Far Side Disk - 2 ",20,0.5,100.5);
  EffGlobDm2 = dbe->book1D("GlobEfficiencyDisk_m2","Efficiency Disk -2",218,0.5,218.5);
  //EffGlobDm2far =dbe->book1D("GlobEfficiencyDisk_m2far","Efficiency Far Side Disk -2",109,0.5,109.5);
  //BXGlobDm2 = dbe->book1D("GlobBXDisk_m2near","BX Near Side Disk -2",109,0.5,109.5);
  //BXGlobDm2far = dbe->book1D("GlobBXDisk_m2far","BX Far Side Disk -2",109,0.5,109.5);
  //MaskedGlobDm2 = dbe->book1D("GlobMaskedDisk_m2near","Masked Near Side Disk -2",109,0.5,109.5);
  //MaskedGlobDm2far = dbe->book1D("GlobMaskedDisk_m2far","Masked Far Side Disk -2",109,0.5,109.5);
  //AverageEffDm2=dbe->book1D("AverageEfficiencyDisk_m2near","Average Efficiency Near Side Disk -2 ",109,0.5,109.5);
  //AverageEffDm2far=dbe->book1D("AverageEfficiencyDisk_m2far","Average Efficiency Far Side Disk -2 ",109,0.5,109.5);
  //NoPredictionDm2=dbe->book1D("NoPredictionDisk_m2near","No Predictions Near Side Disk -2 ",109,0.5,109.5);
  //NoPredictionDm2far=dbe->book1D("NoPredictionDisk_m2far","No Predictions Efficiency Far Side Disk -2 ",109,0.5,109.5);

  dbe->setCurrentFolder(folderPath+"Disk_-3");
  EffDistroDm3=dbe->book1D("EffDistroDisk_m3","Efficiency Distribution Disk - 3 ",20,0.5,100.5);
  //EffDistroDm3far=dbe->book1D("EffDistroDisk_m3far","Efficiency Distribution Far Side Disk - 3 ",20,0.5,100.5);
  EffGlobDm3 = dbe->book1D("GlobEfficiencyDisk_m3","Efficiency Disk -3",218,0.5,218.5);
  //EffGlobDm3far =dbe->book1D("GlobEfficiencyDisk_m3far","Efficiency Far Side Disk -3",109,0.5,109.5);
  //BXGlobDm3 = dbe->book1D("GlobBXDisk_m3near","BX Near Side Disk -3",109,0.5,109.5);
  //BXGlobDm3far = dbe->book1D("GlobBXDisk_m3far","BX Far Side Disk -3",109,0.5,109.5);
  //MaskedGlobDm3 = dbe->book1D("GlobMaskedDisk_m3near","Masked Near Side Disk -3",109,0.5,109.5);
  //MaskedGlobDm3far = dbe->book1D("GlobMaskedDisk_m3far","Masked Far Side Disk -3",109,0.5,109.5);
  //AverageEffDm3=dbe->book1D("AverageEfficiencyDisk_m3near","Average Efficiency Near Side Disk -3 ",109,0.5,109.5);
  //AverageEffDm3far=dbe->book1D("AverageEfficiencyDisk_m3far","Average Efficiency Far Side Disk -3 ",109,0.5,109.5);
  //NoPredictionDm3=dbe->book1D("NoPredictionDisk_m3near","No Predictions Near Side Disk -3 ",109,0.5,109.5);
  //NoPredictionDm3far=dbe->book1D("NoPredictionDisk_m3far","No Predictions Efficiency Far Side Disk -3 ",109,0.5,109.5);

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
  rpcUtils.labelYAxisRoll( Wheel2Summary, 0, 2);

  rpcUtils.labelXAxisSector( Wheel1Summary );
  rpcUtils.labelYAxisRoll( Wheel1Summary, 0, 1);

  rpcUtils.labelXAxisSector( Wheel0Summary );
  rpcUtils.labelYAxisRoll( Wheel0Summary, 0, 0);

  rpcUtils.labelXAxisSector( Wheelm1Summary );
  rpcUtils.labelYAxisRoll( Wheelm1Summary, 0, -1);

  rpcUtils.labelXAxisSector( Wheelm2Summary );
  rpcUtils.labelYAxisRoll( Wheelm2Summary, 0, -2);

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
  rpcUtils.labelYAxisRing(Diskm3Summary, 2);

  rpcUtils.labelXAxisSegment(Diskm2Summary);
  rpcUtils.labelYAxisRing(Diskm2Summary, 2);

  rpcUtils.labelXAxisSegment(Diskm1Summary);
  rpcUtils.labelYAxisRing(Diskm1Summary, 2);

  rpcUtils.labelXAxisSegment(Disk1Summary);
  rpcUtils.labelYAxisRing(Disk1Summary, 2);

  rpcUtils.labelXAxisSegment(Disk2Summary);
  rpcUtils.labelYAxisRing(Disk2Summary, 2);

  rpcUtils.labelXAxisSegment(Disk3Summary);
  rpcUtils.labelYAxisRing(Disk3Summary, 2);
  
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
   
  // std::string label,folder;
 //  folder = folderPath;
//   label = folder + "MuonSegEff/Statistics";
//   if(debug) std::cout<<"Getting statistcs="<<label<<std::endl;
//   statistics = dbe->get(label);
//   if(debug && !statistics){
 
//     std::cout<<"Statistics Doesn't exist Not access to a monitor element"<<std::endl;
//     std::cout<<label<<std::endl;
//     edm::LogWarning("Missing rpcSource in the sequence") << " Statistics Doesn't exist.";
//     return;
//   }
//   if(debug) std::cout<<"Cloning statistcs"<<std::endl;
//   for(int i=1;i<=33;i++){
//     if(debug) std::cout<<statistics->getBinContent(i)<<std::endl;
//     statistics2->setBinContent(i,statistics->getBinContent(i));
//   }
   

   
  //Cloning Residuals.
 
//   folder = folderPath+"MuonSegEff/Residuals/Barrel/";
   
//   label = folder + "GlobalResidualsClu1La1"; hGlobalResClu1La1 = dbe->get(label);
//   label = folder + "GlobalResidualsClu1La2"; hGlobalResClu1La2 = dbe->get(label);
//   label = folder + "GlobalResidualsClu1La3"; hGlobalResClu1La3 = dbe->get(label);
//   label = folder + "GlobalResidualsClu1La4"; hGlobalResClu1La4 = dbe->get(label);
//   label = folder + "GlobalResidualsClu1La5"; hGlobalResClu1La5 = dbe->get(label);
//   label = folder + "GlobalResidualsClu1La6"; hGlobalResClu1La6 = dbe->get(label);

//   label = folder + "GlobalResidualsClu2La1"; hGlobalResClu2La1 = dbe->get(label);
//   label = folder + "GlobalResidualsClu2La2"; hGlobalResClu2La2 = dbe->get(label);
//   label = folder + "GlobalResidualsClu2La3"; hGlobalResClu2La3 = dbe->get(label);
//   label = folder + "GlobalResidualsClu2La4"; hGlobalResClu2La4 = dbe->get(label);
//   label = folder + "GlobalResidualsClu2La5"; hGlobalResClu2La5 = dbe->get(label);
//   label = folder + "GlobalResidualsClu2La6"; hGlobalResClu2La6 = dbe->get(label);

//   label = folder + "GlobalResidualsClu3La1"; hGlobalResClu3La1 = dbe->get(label);
//   label = folder + "GlobalResidualsClu3La2"; hGlobalResClu3La2 = dbe->get(label);
//   label = folder + "GlobalResidualsClu3La3"; hGlobalResClu3La3 = dbe->get(label);
//   label = folder + "GlobalResidualsClu3La4"; hGlobalResClu3La4 = dbe->get(label);
//   label = folder + "GlobalResidualsClu3La5"; hGlobalResClu3La5 = dbe->get(label);
//   label = folder + "GlobalResidualsClu3La6"; hGlobalResClu3La6 = dbe->get(label);
  
  //  if(debug) std::cout<<"Clonning for Barrel"<<std::endl;
   
 //  for(int i=1;i<=101;i++){
//     if(debug) std::cout<<"Global Residual"<<hGlobalResClu1La1->getBinContent(i)<<std::endl;
//     hGlobal2ResClu1La1->setBinContent(i,hGlobalResClu1La1->getBinContent(i));
//     hGlobal2ResClu1La2->setBinContent(i,hGlobalResClu1La2->getBinContent(i));
//     hGlobal2ResClu1La3->setBinContent(i,hGlobalResClu1La3->getBinContent(i));
//     hGlobal2ResClu1La4->setBinContent(i,hGlobalResClu1La4->getBinContent(i));
//     hGlobal2ResClu1La5->setBinContent(i,hGlobalResClu1La5->getBinContent(i));
//     hGlobal2ResClu1La6->setBinContent(i,hGlobalResClu1La6->getBinContent(i));

//     hGlobal2ResClu2La1->setBinContent(i,hGlobalResClu2La1->getBinContent(i));
//     hGlobal2ResClu2La2->setBinContent(i,hGlobalResClu2La2->getBinContent(i));
//     hGlobal2ResClu2La3->setBinContent(i,hGlobalResClu2La3->getBinContent(i));
//     hGlobal2ResClu2La4->setBinContent(i,hGlobalResClu2La4->getBinContent(i));
//     hGlobal2ResClu2La5->setBinContent(i,hGlobalResClu2La5->getBinContent(i));
//     hGlobal2ResClu2La6->setBinContent(i,hGlobalResClu2La6->getBinContent(i));

//     hGlobal2ResClu3La1->setBinContent(i,hGlobalResClu3La1->getBinContent(i));
//     hGlobal2ResClu3La2->setBinContent(i,hGlobalResClu3La2->getBinContent(i));
//     hGlobal2ResClu3La3->setBinContent(i,hGlobalResClu3La3->getBinContent(i));
//     hGlobal2ResClu3La4->setBinContent(i,hGlobalResClu3La4->getBinContent(i));
//     hGlobal2ResClu3La5->setBinContent(i,hGlobalResClu3La5->getBinContent(i));
//     hGlobal2ResClu3La6->setBinContent(i,hGlobalResClu3La6->getBinContent(i));
//   }
 
//   if(debug) std::cout<<"Clonning the EndCap"<<std::endl;
//   folder = folderPath+"MuonSegEff/Residuals/EndCap/";
//   if(debug) std::cout<<folder<<std::endl;
   
//   label = folder + "GlobalResidualsClu1R3C"; hGlobalResClu1R3C = dbe->get(label); 
//   //if(!hGlobalResClu1R3C)std::cout<<"hGlobal2ResClu1R3C dont exist"<<std::endl;
//   //std::cout<<label<<std::endl;
//   label = folder + "GlobalResidualsClu1R3B"; hGlobalResClu1R3B = dbe->get(label);
//   //if(!hGlobalResClu1R3B)std::cout<<"hGlobal2ResClu1R3B dont exist"<<std::endl;
//   //std::cout<<label<<std::endl;
//   label = folder + "GlobalResidualsClu1R3A"; hGlobalResClu1R3A = dbe->get(label); 
//   label = folder + "GlobalResidualsClu1R2C"; hGlobalResClu1R2C = dbe->get(label); 
//   label = folder + "GlobalResidualsClu1R2B"; hGlobalResClu1R2B = dbe->get(label); 
//   label = folder + "GlobalResidualsClu1R2A"; hGlobalResClu1R2A = dbe->get(label); 
  		 								   	   
//   label = folder + "GlobalResidualsClu2R3C"; hGlobalResClu2R3C = dbe->get(label); 
//   label = folder + "GlobalResidualsClu2R3B"; hGlobalResClu2R3B = dbe->get(label); 
//   label = folder + "GlobalResidualsClu2R3A"; hGlobalResClu2R3A = dbe->get(label); 
//   label = folder + "GlobalResidualsClu2R2C"; hGlobalResClu2R2C = dbe->get(label); 
//   label = folder + "GlobalResidualsClu2R2B"; hGlobalResClu2R2B = dbe->get(label); 
//   label = folder + "GlobalResidualsClu2R2A"; hGlobalResClu2R2A = dbe->get(label); 
	 									   	   
//   label = folder + "GlobalResidualsClu3R3C"; hGlobalResClu3R3C = dbe->get(label); 
//   label = folder + "GlobalResidualsClu3R3B"; hGlobalResClu3R3B = dbe->get(label); 
//   label = folder + "GlobalResidualsClu3R3A"; hGlobalResClu3R3A = dbe->get(label); 
//   label = folder + "GlobalResidualsClu3R2C"; hGlobalResClu3R2C = dbe->get(label); 
//   label = folder + "GlobalResidualsClu3R2B"; hGlobalResClu3R2B = dbe->get(label); 
//   label = folder + "GlobalResidualsClu3R2A"; hGlobalResClu3R2A = dbe->get(label); 
 

  // if(debug) std::cout<<"Going for!"<<std::endl;
  /*for(int i=1;i<=101;i++){
    if(debug) std::cout<<i<<std::endl;
    hGlobal2ResClu1R3C->setBinContent(i,hGlobalResClu1R3C->getBinContent(i)); std::cout<<"hGlobal2ResClu1R3C"<<std::endl;
    hGlobal2ResClu1R3B->setBinContent(i,hGlobalResClu1R3B->getBinContent(i)); std::cout<<"hGlobal2ResClu1R3B"<<std::endl;
    hGlobal2ResClu1R3A->setBinContent(i,hGlobalResClu1R3A->getBinContent(i)); std::cout<<"hGlobal2ResClu1R3A"<<std::endl;
    hGlobal2ResClu1R2C->setBinContent(i,hGlobalResClu1R2C->getBinContent(i)); std::cout<<"hGlobal2ResClu1R2C"<<std::endl;
    hGlobal2ResClu1R2B->setBinContent(i,hGlobalResClu1R2B->getBinContent(i)); std::cout<<"hGlobal2ResClu1R2B"<<std::endl;
    hGlobal2ResClu1R2A->setBinContent(i,hGlobalResClu1R2A->getBinContent(i)); std::cout<<"hGlobal2ResClu1R2A"<<std::endl;
    		      							      				    	       
    hGlobal2ResClu2R3C->setBinContent(i,hGlobalResClu2R3C->getBinContent(i)); std::cout<<"hGlobal2ResClu2R3C"<<std::endl;
    hGlobal2ResClu2R3B->setBinContent(i,hGlobalResClu2R3B->getBinContent(i)); std::cout<<"hGlobal2ResClu2R3B"<<std::endl;
    hGlobal2ResClu2R3A->setBinContent(i,hGlobalResClu2R3A->getBinContent(i)); std::cout<<"hGlobal2ResClu2R3A"<<std::endl;
    hGlobal2ResClu2R2C->setBinContent(i,hGlobalResClu2R2C->getBinContent(i)); std::cout<<"hGlobal2ResClu2R2C"<<std::endl;
    hGlobal2ResClu2R2B->setBinContent(i,hGlobalResClu2R2B->getBinContent(i)); std::cout<<"hGlobal2ResClu2R2B"<<std::endl;
    hGlobal2ResClu2R2A->setBinContent(i,hGlobalResClu2R2A->getBinContent(i)); std::cout<<"hGlobal2ResClu2R2A"<<std::endl;
    		      							      				    	       
    hGlobal2ResClu3R3C->setBinContent(i,hGlobalResClu3R3C->getBinContent(i)); std::cout<<"hGlobal2ResClu3R3C"<<std::endl;
    hGlobal2ResClu3R3B->setBinContent(i,hGlobalResClu3R3B->getBinContent(i)); std::cout<<"hGlobal2ResClu3R3B"<<std::endl;
    hGlobal2ResClu3R3A->setBinContent(i,hGlobalResClu3R3A->getBinContent(i)); std::cout<<"hGlobal2ResClu3R3A"<<std::endl;
    hGlobal2ResClu3R2C->setBinContent(i,hGlobalResClu3R2C->getBinContent(i)); std::cout<<"hGlobal2ResClu3R2C"<<std::endl;
    hGlobal2ResClu3R2B->setBinContent(i,hGlobalResClu3R2B->getBinContent(i)); std::cout<<"hGlobal2ResClu3R2B"<<std::endl;
    hGlobal2ResClu3R2A->setBinContent(i,hGlobalResClu3R2A->getBinContent(i)); std::cout<<"hGlobal2ResClu3R2A"<<std::endl;
    }*/
   
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

//   for(TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
//     if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
//       RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
//       std::vector< const RPCRoll*> roles = (ch->rolls());
//       for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
// 	RPCDetId rpcId = (*r)->id();
// 	RPCGeomServ rpcsrv(rpcId);
// 	std::string nameRoll = rpcsrv.name();
// 	if(debug) std::cout<<"Booking for "<<nameRoll<<std::endl;
// 	meCollection[rpcId.rawId()] = bookDetUnitSeg(rpcId,(*r)->nstrips(),folderPath);
//       }
//     }
//   }

  //if(debug) dbe->showDirStructure();

  for(TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if(dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	RPCGeomServ rpcsrv(rpcId);
//	int sector = rpcId.sector();	
	std::string nameRoll = rpcsrv.name();
	// std::map<std::string, MonitorElement*> meMap=meCollection[rpcId.rawId()];
// 	if(debug){
// 	  std::map<std::string, MonitorElement*>::const_iterator it;
// 	  for (it = meMap.begin(); it != meMap.end(); ++it){
// 	    std::cout<<"Histo name:" <<it->first<<std::endl;
// 	  }
// 	}
	 

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
	  // std::string      meIdPRO, 
	  // std::string meIdRPC2, meIdDT2; //, bxDistroId2;
	  
	  //  RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
	  
	  // std::string folder = folderPath+"MuonSegEff";

	  //if(debug) std::cout<<"Setting the folder "<<folder<<std::endl;

	  meIdRPC<<folderPath<<"MuonSegEff/RPCDataOccupancyFromDT_"<<rpcId.rawId();
	  meIdDT<<folderPath<<"MuonSegEff/ExpectedOccupancyFromDT_"<<rpcId.rawId();
	  //	  bxDistroId<<folder<<"/BXDistribution_"<<rpcId.rawId();
      
	  // std::string folder2 = folderPath+"RollByRoll/" +  folderStr->folderStructure(rpcId); 

	  // delete folderStr;

	 //  meIdRPC2 = "RPCDataOccupancyFromDT_" + rpcsrv.name();	
// 	  meIdDT2 =  "ExpectedOccupancyFromDT_"+ rpcsrv.name();
	  
	  // meIdPRO = "Profile_"+ rpcsrv.name();
	  
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
	  float averageerr = 0.;
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
	      averageerr = sqrt(sumbuffer/float(NumberStripsPointed))*100.;
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

//	      BXGlobWm2->setBinContent(indexWheel[0],mybxhisto);  
// 	      BXGlobWm2->setBinError(indexWheel[0],mybxerror);  
// 	      BXGlobWm2->setBinLabel(indexWheel[0],camera,1);
	      
//	      MaskedGlobWm2->setBinContent(indexWheel[0],maskedratio);  
//	      MaskedGlobWm2->setBinLabel(indexWheel[0],camera,1);

//	      AverageEffWm2->setBinContent(indexWheel[0],averageeff);
//	      AverageEffWm2->setBinError(indexWheel[0],averageerr);  
//	      AverageEffWm2->setBinLabel(indexWheel[0],camera,1);
	      
//	      NoPredictionWm2->setBinContent(indexWheel[0],nopredictionsratio);
//            NoPredictionWm2->setBinLabel(indexWheel[0],camera,1);
	    }else if(Ring==-1){
	      EffDistroWm1->Fill(averageeff);
	      indexWheel[1]++;  
	      EffGlobWm1->setBinContent(indexWheel[1],ef);  
	      EffGlobWm1->setBinError(indexWheel[1],er);  
	      EffGlobWm1->setBinLabel(indexWheel[1],camera,1);  
	      
// 	      BXGlobWm1->setBinContent(indexWheel[1],mybxhisto);  
// 	      BXGlobWm1->setBinError(indexWheel[1],mybxerror);  
// 	      BXGlobWm1->setBinLabel(indexWheel[1],camera,1);
	      
//	      MaskedGlobWm1->setBinContent(indexWheel[1],maskedratio);  
//	      MaskedGlobWm1->setBinLabel(indexWheel[1],camera,1);

//	      AverageEffWm1->setBinContent(indexWheel[1],averageeff);
//	      AverageEffWm1->setBinError(indexWheel[1],averageerr);  
//	      AverageEffWm1->setBinLabel(indexWheel[1],camera,1);
	      
//	      NoPredictionWm1->setBinContent(indexWheel[1],nopredictionsratio);
//            NoPredictionWm1->setBinLabel(indexWheel[1],camera,1);

	    }else if(Ring==0){
	      EffDistroW0->Fill(averageeff);
	      indexWheel[2]++;  
	      EffGlobW0->setBinContent(indexWheel[2],ef);  
	      EffGlobW0->setBinError(indexWheel[2],er);  
	      EffGlobW0->setBinLabel(indexWheel[2],camera,1);  
	      
// 	      BXGlobW0->setBinContent(indexWheel[2],mybxhisto);  
// 	      BXGlobW0->setBinError(indexWheel[2],mybxerror);  
// 	      BXGlobW0->setBinLabel(indexWheel[2],camera,1);

//	      MaskedGlobW0->setBinContent(indexWheel[2],maskedratio);  
//	      MaskedGlobW0->setBinLabel(indexWheel[2],camera,1);
	      
//	      AverageEffW0->setBinContent(indexWheel[2],averageeff);
//	      AverageEffW0->setBinError(indexWheel[2],averageerr);  
//	      AverageEffW0->setBinLabel(indexWheel[2],camera,1);
	      
//	      NoPredictionW0->setBinContent(indexWheel[2],nopredictionsratio);
//            NoPredictionW0->setBinLabel(indexWheel[2],camera,1);	      
	    }else if(Ring==1){
	      EffDistroW1->Fill(averageeff);
	      indexWheel[3]++;  
	      EffGlobW1->setBinContent(indexWheel[3],ef);  
	      EffGlobW1->setBinError(indexWheel[3],er);  
	      EffGlobW1->setBinLabel(indexWheel[3],camera,1);  
	      
//	      BXGlobW1->setBinContent(indexWheel[3],mybxhisto);  
// 	      BXGlobW1->setBinError(indexWheel[3],mybxerror);  
// 	      BXGlobW1->setBinLabel(indexWheel[3],camera,1);

//	      MaskedGlobW1->setBinContent(indexWheel[3],maskedratio);  
//	      MaskedGlobW1->setBinLabel(indexWheel[3],camera,1);

//	      AverageEffW1->setBinContent(indexWheel[3],averageeff);
//	      AverageEffW1->setBinError(indexWheel[3],averageerr);  
//	      AverageEffW1->setBinLabel(indexWheel[3],camera,1);
	      
//	      NoPredictionW1->setBinContent(indexWheel[3],nopredictionsratio);
//            NoPredictionW1->setBinLabel(indexWheel[3],camera,1);	      
	    }else if(Ring==2){
	      EffDistroW2->Fill(averageeff);
	      indexWheel[4]++;
	      EffGlobW2->setBinContent(indexWheel[4],ef);
	      EffGlobW2->setBinError(indexWheel[4],er);
	      EffGlobW2->setBinLabel(indexWheel[4],camera,1);

//	      BXGlobW2->setBinContent(indexWheel[4],mybxhisto);  
// 	      BXGlobW2->setBinError(indexWheel[4],mybxerror);  
// 	      BXGlobW2->setBinLabel(indexWheel[4],camera,1);
	      
//	      MaskedGlobW2->setBinContent(indexWheel[4],maskedratio);  
//	      MaskedGlobW2->setBinLabel(indexWheel[4],camera,1);

//	      AverageEffW2->setBinContent(indexWheel[4],averageeff);
//	      AverageEffW2->setBinError(indexWheel[4],averageerr);  
//	      AverageEffW2->setBinLabel(indexWheel[4],camera,1);
	      
//	      NoPredictionW2->setBinContent(indexWheel[4],nopredictionsratio);
//              NoPredictionW2->setBinLabel(indexWheel[4],camera,1);	      
	    }
/*	  }else{//Far Side 
	    if(Ring==-2){
	      EffDistroWm2far->Fill(averageeff);
	      indexWheelf[0]++;  
	      EffGlobWm2far->setBinContent(indexWheelf[0],ef);  
	      EffGlobWm2far->setBinError(indexWheelf[0],er);  
	      EffGlobWm2far->setBinLabel(indexWheelf[0],camera,1);

// 	      BXGlobWm2far->setBinContent(indexWheelf[0],mybxhisto);  
// 	      BXGlobWm2far->setBinError(indexWheelf[0],mybxerror);  
// 	      BXGlobWm2far->setBinLabel(indexWheelf[0],camera);
	      
	      MaskedGlobWm2far->setBinContent(indexWheelf[0],maskedratio);
	      MaskedGlobWm2far->setBinLabel(indexWheelf[0],camera,1);
	      
	      AverageEffWm2far->setBinContent(indexWheelf[0],averageeff);
              AverageEffWm2far->setBinError(indexWheelf[0],averageerr);
              AverageEffWm2far->setBinLabel(indexWheelf[0],camera,1);

              NoPredictionWm2->setBinContent(indexWheel[0],nopredictionsratio);
              NoPredictionWm2->setBinLabel(indexWheel[0],camera,1);

	    }else if(Ring==-1){
	      EffDistroWm1far->Fill(averageeff);
	      indexWheelf[1]++;  
	      EffGlobWm1far->setBinContent(indexWheelf[1],ef);  
	      EffGlobWm1far->setBinError(indexWheelf[1],er);  
	      EffGlobWm1far->setBinLabel(indexWheelf[1],camera,1);  
	      
	  //     BXGlobWm1far->setBinContent(indexWheelf[1],mybxhisto);  
// 	      BXGlobWm1far->setBinError(indexWheelf[1],mybxerror);  
// 	      BXGlobWm1far->setBinLabel(indexWheelf[1],camera,1);
	      
	      MaskedGlobWm1far->setBinContent(indexWheelf[1],maskedratio);
	      MaskedGlobWm1far->setBinLabel(indexWheelf[1],camera,1);

	      AverageEffWm1far->setBinContent(indexWheelf[1],averageeff);
              AverageEffWm1far->setBinError(indexWheelf[1],averageerr);
              AverageEffWm1far->setBinLabel(indexWheelf[1],camera,1);

              NoPredictionWm1far->setBinContent(indexWheelf[1],nopredictionsratio);
              NoPredictionWm1far->setBinLabel(indexWheelf[1],camera,1);

	    }else  if(Ring==0){
	      EffDistroW0far->Fill(averageeff);
	      indexWheelf[2]++;  
	      EffGlobW0far->setBinContent(indexWheelf[2],ef);  
	      EffGlobW0far->setBinError(indexWheelf[2],er);  
	      EffGlobW0far->setBinLabel(indexWheelf[2],camera,1);  
	      
// 	      BXGlobW0far->setBinContent(indexWheelf[2],mybxhisto);  
// 	      BXGlobW0far->setBinError(indexWheelf[2],mybxerror);  
// 	      BXGlobW0far->setBinLabel(indexWheelf[2],camera,1);

	      MaskedGlobW0far->setBinContent(indexWheelf[2],maskedratio);
	      MaskedGlobW0far->setBinLabel(indexWheelf[2],camera,1);

	      AverageEffW0far->setBinContent(indexWheelf[2],averageeff);
              AverageEffW0far->setBinError(indexWheelf[2],averageerr);
              AverageEffW0far->setBinLabel(indexWheelf[2],camera,1);

              NoPredictionW0far->setBinContent(indexWheelf[2],nopredictionsratio);
              NoPredictionW0far->setBinLabel(indexWheelf[2],camera,1);
	    }else if(Ring==1){
	      EffDistroW1far->Fill(averageeff);
	      indexWheelf[3]++;  
	      EffGlobW1far->setBinContent(indexWheelf[3],ef);  
	      EffGlobW1far->setBinError(indexWheelf[3],er);  
	      EffGlobW1far->setBinLabel(indexWheelf[3],camera,1);  
	      
	//       BXGlobW1far->setBinContent(indexWheelf[3],mybxhisto);  
// 	      BXGlobW1far->setBinError(indexWheelf[3],mybxerror);  
// 	      BXGlobW1far->setBinLabel(indexWheelf[3],camera,1);

	      MaskedGlobW1far->setBinContent(indexWheelf[3],maskedratio);
	      MaskedGlobW1far->setBinLabel(indexWheelf[3],camera,1);
	      
	      AverageEffW1far->setBinContent(indexWheelf[3],averageeff);
              AverageEffW1far->setBinError(indexWheelf[3],averageerr);
              AverageEffW1far->setBinLabel(indexWheelf[3],camera,1);

              NoPredictionW1far->setBinContent(indexWheelf[3],nopredictionsratio);
              NoPredictionW1far->setBinLabel(indexWheelf[3],camera,1);

	    }else if(Ring==2){
	      EffDistroW2far->Fill(averageeff);
	      indexWheelf[4]++;
	      EffGlobW2far->setBinContent(indexWheelf[4],ef);
	      EffGlobW2far->setBinError(indexWheelf[4],er);
	      EffGlobW2far->setBinLabel(indexWheelf[4],camera,1);

	 //      BXGlobW2far->setBinContent(indexWheelf[4],mybxhisto);  
// 	      BXGlobW2far->setBinError(indexWheelf[4],mybxerror);  
// 	      BXGlobW2far->setBinLabel(indexWheelf[4],camera,1);
	      
	      MaskedGlobW2far->setBinContent(indexWheelf[4],maskedratio);
	      MaskedGlobW2far->setBinLabel(indexWheelf[4],camera,1);

	      AverageEffW2far->setBinContent(indexWheelf[4],averageeff);
              AverageEffW2far->setBinError(indexWheelf[4],averageerr);
              AverageEffW2far->setBinLabel(indexWheelf[4],camera,1);

              NoPredictionW2far->setBinContent(indexWheelf[4],nopredictionsratio);
              NoPredictionW2far->setBinLabel(indexWheelf[4],camera,1);
	    }
	  }*/
	}else{//EndCap

	  std::stringstream meIdRPC,meIdCSC; //, bxDistroId;
	  std::string      meIdPRO;
	  // std::string   meIdRPC2, meIdCSC2; //, bxDistroId2;
	  
	  // RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
	 
	  //  std::string folder = folderPath+"MuonSegEff";
	  
	  meIdRPC<<folderPath<<"MuonSegEff/RPCDataOccupancyFromCSC_"<<rpcId.rawId();
	  meIdCSC<<folderPath<<"MuonSegEff/ExpectedOccupancyFromCSC_"<<rpcId.rawId();
	  //  bxDistroId<<folder<<"/BXDistribution_"<<rpcId.rawId();
		
	  // std::string folder2 = folderPath+"RollByRoll/" +  folderStr->folderStructure(rpcId); 
	  
	  // delete folderStr;

// 	  meIdRPC2 = "RPCDataOccupancyFromCSC_" + rpcsrv.name();	
// 	  meIdCSC2 =  "ExpectedOccupancyFromCSC_"+ rpcsrv.name();
	  //	  bxDistroId2 = "BXDistribution_"+ rpcsrv.name();

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
	  float averageerr = 0;
	  int NumberStripsPointed = 0;

	  //	  if(debug) std::cout<<"Cloning BX"<<std::endl;

// 	  for(int i=1;i<=11;i++){
// 	    if(debug) std::cout<<i<<"-"<<BXDistribution->getBinContent(i)<<std::endl;
// 	    meMap[bxDistroId2]->setBinContent(i,BXDistribution->getBinContent(i));
// 	  }

	  if(histoRPC && histoCSC) {// && BXDistribution){
	    if(debug) std::cout <<rpcsrv.name()<<std::endl;
	    
	    for(int i=1;i<=int((*r)->nstrips());++i){
// 	      if(debug) std::cout<<"Cloning histoCSC "<<meIdCSC2<<std::endl;
// 	      meMap[meIdCSC2]->setBinContent(i,histoCSC->getBinContent(i));
// 	      if(debug) std::cout<<"Cloning histoRPC:"<<meIdRPC2<<std::endl;
// 	      meMap[meIdRPC2]->setBinContent(i,histoRPC->getBinContent(i));
	      
// 	      if(meMap.find(meIdPRO)==meMap.end()){
// 		std::cout<<"Empty Map"<<std::endl;
// 	      }

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
	      averageerr = sqrt(sumbuffer/float(NumberStripsPointed))*100.;
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

//	      BXGlobDm3->setBinContent(indexDisk[0],mybxhisto);  
// 	      BXGlobDm3->setBinError(indexDisk[0],mybxerror);  
// 	      BXGlobDm3->setBinLabel(indexDisk[0],camera,1);
	      
//	      MaskedGlobDm3->setBinContent(indexDisk[0],maskedratio);  
//	      MaskedGlobDm3->setBinLabel(indexDisk[0],camera,1);

//	      AverageEffDm3->setBinContent(indexDisk[0],averageeff);
//	      AverageEffDm3->setBinError(indexDisk[0],averageerr);  
//	      AverageEffDm3->setBinLabel(indexDisk[0],camera,1);
	      
//	      NoPredictionDm3->setBinContent(indexDisk[0],nopredictionsratio);
//            NoPredictionDm3->setBinLabel(indexDisk[0],camera,1);
	    }else if(Disk==-2){
	      EffDistroDm2->Fill(averageeff);
	      indexDisk[1]++;  
	      EffGlobDm2->setBinContent(indexDisk[1],ef);  
	      EffGlobDm2->setBinError(indexDisk[1],er);  
	      EffGlobDm2->setBinLabel(indexDisk[1],camera,1);

// 	      BXGlobDm2->setBinContent(indexDisk[1],mybxhisto);  
// 	      BXGlobDm2->setBinError(indexDisk[1],mybxerror);  
// 	      BXGlobDm2->setBinLabel(indexDisk[1],camera,1);
	      
//	      MaskedGlobDm2->setBinContent(indexDisk[1],maskedratio);  
//	      MaskedGlobDm2->setBinLabel(indexDisk[1],camera,1);

//	      AverageEffDm2->setBinContent(indexDisk[1],averageeff);
//	      AverageEffDm2->setBinError(indexDisk[1],averageerr);  
//	      AverageEffDm2->setBinLabel(indexDisk[1],camera,1);
	      
//	      NoPredictionDm2->setBinContent(indexDisk[1],nopredictionsratio);
//            NoPredictionDm2->setBinLabel(indexDisk[1],camera,1);
	    }else if(Disk==-1){
	      EffDistroDm1->Fill(averageeff);
	      indexDisk[2]++;  
	      EffGlobDm1->setBinContent(indexDisk[2],ef);  
	      EffGlobDm1->setBinError(indexDisk[2],er);  
	      EffGlobDm1->setBinLabel(indexDisk[2],camera,1);  
	      
// 	      BXGlobDm1->setBinContent(indexDisk[2],mybxhisto);  
// 	      BXGlobDm1->setBinError(indexDisk[2],mybxerror);  
// 	      BXGlobDm1->setBinLabel(indexDisk[2],camera,1);
	      
//	      MaskedGlobDm1->setBinContent(indexDisk[2],maskedratio);  
//	      MaskedGlobDm1->setBinLabel(indexDisk[2],camera,1);

//	      AverageEffDm1->setBinContent(indexDisk[2],averageeff);
//	      AverageEffDm1->setBinError(indexDisk[2],averageerr);  
//	      AverageEffDm1->setBinLabel(indexDisk[2],camera,1);
	      
//	      NoPredictionDm1->setBinContent(indexDisk[2],nopredictionsratio);
//            NoPredictionDm1->setBinLabel(indexDisk[2],camera,1);

	    }else if(Disk==1){
	      EffDistroD1->Fill(averageeff);
	      indexDisk[3]++;  
	      EffGlobD1->setBinContent(indexDisk[3],ef);  
	      EffGlobD1->setBinError(indexDisk[3],er);  
	      EffGlobD1->setBinLabel(indexDisk[3],camera,1);  
	      
// 	      BXGlobD1->setBinContent(indexDisk[3],mybxhisto);  
// 	      BXGlobD1->setBinError(indexDisk[3],mybxerror);  
// 	      BXGlobD1->setBinLabel(indexDisk[3],camera,1);

//	      MaskedGlobD1->setBinContent(indexDisk[3],maskedratio);  
//	      MaskedGlobD1->setBinLabel(indexDisk[3],camera,1);

//	      AverageEffD1->setBinContent(indexDisk[3],averageeff);
//	      AverageEffD1->setBinError(indexDisk[3],averageerr);  
//	      AverageEffD1->setBinLabel(indexDisk[3],camera,1);
	      
//	      NoPredictionD1->setBinContent(indexDisk[3],nopredictionsratio);
//            NoPredictionD1->setBinLabel(indexDisk[3],camera,1);	      
	    }else if(Disk==2){
	      EffDistroD2->Fill(averageeff);
	      indexDisk[4]++;
	      EffGlobD2->setBinContent(indexDisk[4],ef);
	      EffGlobD2->setBinError(indexDisk[4],er);
	      EffGlobD2->setBinLabel(indexDisk[4],camera,1);

//	      BXGlobD2->setBinContent(indexDisk[4],mybxhisto);  
// 	      BXGlobD2->setBinError(indexDisk[4],mybxerror);  
// 	      BXGlobD2->setBinLabel(indexDisk[4],camera,1);
	      
//	      MaskedGlobD2->setBinContent(indexDisk[4],maskedratio);  
//	      MaskedGlobD2->setBinLabel(indexDisk[4],camera,1);

//	      AverageEffD2->setBinContent(indexDisk[4],averageeff);
//	      AverageEffD2->setBinError(indexDisk[4],averageerr);  
//	      AverageEffD2->setBinLabel(indexDisk[4],camera,1);
	      
//	      NoPredictionD2->setBinContent(indexDisk[4],nopredictionsratio);
//            NoPredictionD2->setBinLabel(indexDisk[4],camera,1);	      
	    }else if(Disk==3){
	      EffDistroD3->Fill(averageeff);
	      indexDisk[5]++;
	      EffGlobD3->setBinContent(indexDisk[5],ef);
	      EffGlobD3->setBinError(indexDisk[5],er);
	      EffGlobD3->setBinLabel(indexDisk[5],camera,1);

// 	      BXGlobD3->setBinContent(indexDisk[5],mybxhisto);  
// 	      BXGlobD3->setBinError(indexDisk[5],mybxerror);  
// 	      BXGlobD3->setBinLabel(indexDisk[5],camera,1);
	      
//	      MaskedGlobD3->setBinContent(indexDisk[5],maskedratio);  
//	      MaskedGlobD3->setBinLabel(indexDisk[5],camera,1);

//	      AverageEffD3->setBinContent(indexDisk[5],averageeff);
//	      AverageEffD3->setBinError(indexDisk[5],averageerr);  
//	      AverageEffD3->setBinLabel(indexDisk[5],camera,1);
	      
//	      NoPredictionD3->setBinContent(indexDisk[5],nopredictionsratio);
//            NoPredictionD3->setBinLabel(indexDisk[5],camera,1);	      
	    }
/*	  }else{//Far Side 
	    
	    if(Disk==-3){
	      EffDistroDm3far->Fill(averageeff);
	      indexDiskf[0]++;  
	      EffGlobDm3far->setBinContent(indexDiskf[0],ef);  
	      EffGlobDm3far->setBinError(indexDiskf[0],er);  
	      EffGlobDm3far->setBinLabel(indexDiskf[0],camera,1);

	 //      BXGlobDm3far->setBinContent(indexDiskf[0],mybxhisto);  
// 	      BXGlobDm3far->setBinError(indexDiskf[0],mybxerror);  
// 	      BXGlobDm3far->setBinLabel(indexDiskf[0],camera);
	      
	      MaskedGlobDm3far->setBinContent(indexDiskf[0],maskedratio);
	      MaskedGlobDm3far->setBinLabel(indexDiskf[0],camera,1);
	      
	      AverageEffDm3far->setBinContent(indexDiskf[0],averageeff);
              AverageEffDm3far->setBinError(indexDiskf[0],averageerr);
              AverageEffDm3far->setBinLabel(indexDiskf[0],camera,1);

              NoPredictionDm3->setBinContent(indexDisk[0],nopredictionsratio);
              NoPredictionDm3->setBinLabel(indexDisk[0],camera,1);

	    }
	    else if(Disk==-2){
	      EffDistroDm2far->Fill(averageeff);
	      indexDiskf[1]++;  
	      EffGlobDm2far->setBinContent(indexDiskf[1],ef);  
	      EffGlobDm2far->setBinError(indexDiskf[1],er);  
	      EffGlobDm2far->setBinLabel(indexDiskf[1],camera,1);

	     //  BXGlobDm2far->setBinContent(indexDiskf[1],mybxhisto);  
// 	      BXGlobDm2far->setBinError(indexDiskf[1],mybxerror);  
// 	      BXGlobDm2far->setBinLabel(indexDiskf[1],camera);
	      
	      MaskedGlobDm2far->setBinContent(indexDiskf[1],maskedratio);
	      MaskedGlobDm2far->setBinLabel(indexDiskf[1],camera,1);
	      
	      AverageEffDm2far->setBinContent(indexDiskf[1],averageeff);
              AverageEffDm2far->setBinError(indexDiskf[1],averageerr);
              AverageEffDm2far->setBinLabel(indexDiskf[1],camera,1);

              NoPredictionDm2->setBinContent(indexDisk[1],nopredictionsratio);
              NoPredictionDm2->setBinLabel(indexDisk[1],camera,1);

	    }else if(Disk==-1){
	      EffDistroDm1far->Fill(averageeff);
	      indexDiskf[2]++;  
	      EffGlobDm1far->setBinContent(indexDiskf[2],ef);  
	      EffGlobDm1far->setBinError(indexDiskf[2],er);  
	      EffGlobDm1far->setBinLabel(indexDiskf[2],camera,1);  
	      
// 	      BXGlobDm1far->setBinContent(indexDiskf[2],mybxhisto);  
// 	      BXGlobDm1far->setBinError(indexDiskf[2],mybxerror);  
// 	      BXGlobDm1far->setBinLabel(indexDiskf[2],camera,1);
	      
	      MaskedGlobDm1far->setBinContent(indexDiskf[2],maskedratio);
	      MaskedGlobDm1far->setBinLabel(indexDiskf[2],camera,1);

	      AverageEffDm1far->setBinContent(indexDiskf[2],averageeff);
              AverageEffDm1far->setBinError(indexDiskf[2],averageerr);
              AverageEffDm1far->setBinLabel(indexDiskf[2],camera,1);

              NoPredictionDm1far->setBinContent(indexDiskf[2],nopredictionsratio);
              NoPredictionDm1far->setBinLabel(indexDiskf[2],camera,1);

	    }else if(Disk==1){
	      EffDistroD1far->Fill(averageeff);
	      indexDiskf[3]++;  
	      EffGlobD1far->setBinContent(indexDiskf[3],ef);  
	      EffGlobD1far->setBinError(indexDiskf[3],er);  
	      EffGlobD1far->setBinLabel(indexDiskf[3],camera,1);  
	      
// 	      BXGlobD1far->setBinContent(indexDiskf[3],mybxhisto);  
// 	      BXGlobD1far->setBinError(indexDiskf[3],mybxerror);  
// 	      BXGlobD1far->setBinLabel(indexDiskf[3],camera,1);

	      MaskedGlobD1far->setBinContent(indexDiskf[3],maskedratio);
	      MaskedGlobD1far->setBinLabel(indexDiskf[3],camera,1);
	      
	      AverageEffD1far->setBinContent(indexDiskf[3],averageeff);
              AverageEffD1far->setBinError(indexDiskf[3],averageerr);
              AverageEffD1far->setBinLabel(indexDiskf[3],camera,1);

              NoPredictionD1far->setBinContent(indexDiskf[3],nopredictionsratio);
              NoPredictionD1far->setBinLabel(indexDiskf[3],camera,1);

	    }else if(Disk==2){
	      EffDistroD2far->Fill(averageeff);
	      indexDiskf[4]++;
	      EffGlobD2far->setBinContent(indexDiskf[4],ef);
	      EffGlobD2far->setBinError(indexDiskf[4],er);
	      EffGlobD2far->setBinLabel(indexDiskf[4],camera,1);

// 	      BXGlobD2far->setBinContent(indexDiskf[4],mybxhisto);  
// 	      BXGlobD2far->setBinError(indexDiskf[4],mybxerror);  
// 	      BXGlobD2far->setBinLabel(indexDiskf[4],camera,1);
	      
	      MaskedGlobD2far->setBinContent(indexDiskf[4],maskedratio);
	      MaskedGlobD2far->setBinLabel(indexDiskf[4],camera,1);

	      AverageEffD2far->setBinContent(indexDiskf[4],averageeff);
              AverageEffD2far->setBinError(indexDiskf[4],averageerr);
              AverageEffD2far->setBinLabel(indexDiskf[4],camera,1);

              NoPredictionD2far->setBinContent(indexDiskf[4],nopredictionsratio);
              NoPredictionD2far->setBinLabel(indexDiskf[4],camera,1);
	    }else if(Disk==3){
	      EffDistroD3far->Fill(averageeff);
	      indexDiskf[5]++;
	      EffGlobD3far->setBinContent(indexDiskf[5],ef);
	      EffGlobD3far->setBinError(indexDiskf[5],er);
	      EffGlobD3far->setBinLabel(indexDiskf[5],camera,1);

// 	      BXGlobD3far->setBinContent(indexDiskf[5],mybxhisto);  
// 	      BXGlobD3far->setBinError(indexDiskf[5],mybxerror);  
// 	      BXGlobD3far->setBinLabel(indexDiskf[5],camera,1);
	      
	      MaskedGlobD3far->setBinContent(indexDiskf[5],maskedratio);
	      MaskedGlobD3far->setBinLabel(indexDiskf[5],camera,1);

	      AverageEffD3far->setBinContent(indexDiskf[5],averageeff);
              AverageEffD3far->setBinError(indexDiskf[5],averageerr);
              AverageEffD3far->setBinLabel(indexDiskf[5],camera,1);

              NoPredictionD3far->setBinContent(indexDiskf[5],nopredictionsratio);
              NoPredictionD3far->setBinLabel(indexDiskf[5],camera,1);
	    }
	  }//Finishing EndCap*/
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
  
//    EffGlobWm2far->setAxisRange(-4.,100.,2);
//    EffGlobWm1far->setAxisRange(-4.,100.,2);
//    EffGlobW0far->setAxisRange(-4.,100.,2);
//    EffGlobW1far->setAxisRange(-4.,100.,2);
//    EffGlobW2far->setAxisRange(-4.,100.,2);

//    AverageEffWm2->setAxisRange(-4.,100.,2);
//    AverageEffWm1->setAxisRange(-4.,100.,2);
//    AverageEffW0->setAxisRange(-4.,100.,2);
//    AverageEffW1->setAxisRange(-4.,100.,2);
//    AverageEffW2->setAxisRange(-4.,100.,2);
  
//    AverageEffWm2far->setAxisRange(-4.,100.,2);
//    AverageEffWm1far->setAxisRange(-4.,100.,2);
//    AverageEffW0far->setAxisRange(-4.,100.,2);
//    AverageEffW1far->setAxisRange(-4.,100.,2);
//    AverageEffW2far->setAxisRange(-4.,100.,2);

//    MaskedGlobWm2->setAxisRange(-4.,100.,2);
//    MaskedGlobWm1->setAxisRange(-4.,100.,2);
//    MaskedGlobW0->setAxisRange(-4.,100.,2);
//    MaskedGlobW1->setAxisRange(-4.,100.,2);
//    MaskedGlobW2->setAxisRange(-4.,100.,2);
  
//    MaskedGlobWm2far->setAxisRange(-4.,100.,2);
//    MaskedGlobWm1far->setAxisRange(-4.,100.,2);
//    MaskedGlobW0far->setAxisRange(-4.,100.,2);
//    MaskedGlobW1far->setAxisRange(-4.,100.,2);
//    MaskedGlobW2far->setAxisRange(-4.,100.,2);

//    NoPredictionWm2->setAxisRange(-4.,100.,2);
//    NoPredictionWm1->setAxisRange(-4.,100.,2);
//    NoPredictionW0->setAxisRange(-4.,100.,2);
//    NoPredictionW1->setAxisRange(-4.,100.,2);
//    NoPredictionW2->setAxisRange(-4.,100.,2);
  
//    NoPredictionWm2far->setAxisRange(-4.,100.,2);
//    NoPredictionWm1far->setAxisRange(-4.,100.,2);
//    NoPredictionW0far->setAxisRange(-4.,100.,2);
//    NoPredictionW1far->setAxisRange(-4.,100.,2);
//    NoPredictionW2far->setAxisRange(-4.,100.,2);
  }
  //EndCap

  if(endcap){
    EffGlobDm3->setAxisRange(-4.,100.,2);
    EffGlobDm2->setAxisRange(-4.,100.,2);
    EffGlobDm1->setAxisRange(-4.,100.,2);
    EffGlobD1->setAxisRange(-4.,100.,2);
    EffGlobD2->setAxisRange(-4.,100.,2);
    EffGlobD3->setAxisRange(-4.,100.,2);

//    EffGlobDm3far->setAxisRange(-4.,100.,2);
//    EffGlobDm2far->setAxisRange(-4.,100.,2);
//    EffGlobDm1far->setAxisRange(-4.,100.,2);
//    EffGlobD1far->setAxisRange(-4.,100.,2);
//    EffGlobD2far->setAxisRange(-4.,100.,2);
//    EffGlobD3far->setAxisRange(-4.,100.,2);

//     BXGlobDm3->setAxisRange(-4.,100.,2);
//     BXGlobDm2->setAxisRange(-4.,100.,2);
//     BXGlobDm1->setAxisRange(-4.,100.,2);
//     BXGlobD1->setAxisRange(-4.,100.,2);
//     BXGlobD2->setAxisRange(-4.,100.,2);
//     BXGlobD3->setAxisRange(-4.,100.,2);
  
//     BXGlobDm3far->setAxisRange(-4.,100.,2);
//     BXGlobDm2far->setAxisRange(-4.,100.,2);
//     BXGlobDm1far->setAxisRange(-4.,100.,2);
//     BXGlobD1far->setAxisRange(-4.,100.,2);
//     BXGlobD2far->setAxisRange(-4.,100.,2);
//     BXGlobD3far->setAxisRange(-4.,100.,2);

//    MaskedGlobDm3->setAxisRange(-4.,100.,2);
//    MaskedGlobDm2->setAxisRange(-4.,100.,2);
//    MaskedGlobDm1->setAxisRange(-4.,100.,2);
//    MaskedGlobD1->setAxisRange(-4.,100.,2);
//    MaskedGlobD2->setAxisRange(-4.,100.,2);
//    MaskedGlobD3->setAxisRange(-4.,100.,2);
  
//    MaskedGlobDm3far->setAxisRange(-4.,100.,2);
//    MaskedGlobDm2far->setAxisRange(-4.,100.,2);
//    MaskedGlobDm1far->setAxisRange(-4.,100.,2);
//    MaskedGlobD1far->setAxisRange(-4.,100.,2);
//    MaskedGlobD2far->setAxisRange(-4.,100.,2);
//    MaskedGlobD3far->setAxisRange(-4.,100.,2);

//    AverageEffDm3->setAxisRange(-4.,100.,2);
//    AverageEffDm2->setAxisRange(-4.,100.,2);
//    AverageEffDm1->setAxisRange(-4.,100.,2);
//    AverageEffD1->setAxisRange(-4.,100.,2);
//    AverageEffD2->setAxisRange(-4.,100.,2);
//    AverageEffD3->setAxisRange(-4.,100.,2);

//    AverageEffDm3far->setAxisRange(-4.,100.,2);
//    AverageEffDm2far->setAxisRange(-4.,100.,2);
//    AverageEffDm1far->setAxisRange(-4.,100.,2);
//    AverageEffD1far->setAxisRange(-4.,100.,2);
//    AverageEffD2far->setAxisRange(-4.,100.,2);
//    AverageEffD3far->setAxisRange(-4.,100.,2);

//    NoPredictionDm3->setAxisRange(-4.,100.,2);
//    NoPredictionDm2->setAxisRange(-4.,100.,2);
//    NoPredictionDm1->setAxisRange(-4.,100.,2);
//    NoPredictionD1->setAxisRange(-4.,100.,2);
//    NoPredictionD2->setAxisRange(-4.,100.,2);
//    NoPredictionD3->setAxisRange(-4.,100.,2);

//    NoPredictionDm3far->setAxisRange(-4.,100.,2);
//    NoPredictionDm2far->setAxisRange(-4.,100.,2);
//    NoPredictionDm1far->setAxisRange(-4.,100.,2);
//    NoPredictionD1far->setAxisRange(-4.,100.,2);
//    NoPredictionD2far->setAxisRange(-4.,100.,2);
//    NoPredictionD3far->setAxisRange(-4.,100.,2);
  }

  //Title for Both

  //Barrel
  if(barrel){
    EffGlobWm2->setAxisTitle("%",2);
    EffGlobWm1->setAxisTitle("%",2);
    EffGlobW0->setAxisTitle("%",2);
    EffGlobW1->setAxisTitle("%",2);
    EffGlobW2->setAxisTitle("%",2);
  
//    EffGlobWm2far->setAxisTitle("%",2);
//    EffGlobWm1far->setAxisTitle("%",2);
//    EffGlobW0far->setAxisTitle("%",2);
//    EffGlobW1far->setAxisTitle("%",2);
//    EffGlobW2far->setAxisTitle("%",2);

//    AverageEffWm2->setAxisTitle("%",2);
//    AverageEffWm1->setAxisTitle("%",2);
//    AverageEffW0->setAxisTitle("%",2);
//    AverageEffW1->setAxisTitle("%",2);
//    AverageEffW2->setAxisTitle("%",2);
  
//    AverageEffWm2far->setAxisTitle("%",2);
//    AverageEffWm1far->setAxisTitle("%",2);
//    AverageEffW0far->setAxisTitle("%",2);
//    AverageEffW1far->setAxisTitle("%",2);
//    AverageEffW2far->setAxisTitle("%",2);

//    MaskedGlobWm2->setAxisTitle("%",2);
//    MaskedGlobWm1->setAxisTitle("%",2);
//    MaskedGlobW0->setAxisTitle("%",2);
//    MaskedGlobW1->setAxisTitle("%",2);
//    MaskedGlobW2->setAxisTitle("%",2);
  
//    MaskedGlobWm2far->setAxisTitle("%",2);
//    MaskedGlobWm1far->setAxisTitle("%",2);
//    MaskedGlobW0far->setAxisTitle("%",2);
//    MaskedGlobW1far->setAxisTitle("%",2);
//    MaskedGlobW2far->setAxisTitle("%",2);

//    NoPredictionWm2->setAxisTitle("%",2);
//    NoPredictionWm1->setAxisTitle("%",2);
//    NoPredictionW0->setAxisTitle("%",2);
//    NoPredictionW1->setAxisTitle("%",2);
//    NoPredictionW2->setAxisTitle("%",2);
  
//    NoPredictionWm2far->setAxisTitle("%",2);
//    NoPredictionWm1far->setAxisTitle("%",2);
//    NoPredictionW0far->setAxisTitle("%",2);
//    NoPredictionW1far->setAxisTitle("%",2);
//    NoPredictionW2far->setAxisTitle("%",2);
  }
  //EndCap

  if(endcap){
    EffGlobDm3->setAxisTitle("%",2);
    EffGlobDm2->setAxisTitle("%",2);
    EffGlobDm1->setAxisTitle("%",2);
    EffGlobD1->setAxisTitle("%",2);
    EffGlobD2->setAxisTitle("%",2);
    EffGlobD3->setAxisTitle("%",2);

//    EffGlobDm3far->setAxisTitle("%",2);
//    EffGlobDm2far->setAxisTitle("%",2);
//    EffGlobDm1far->setAxisTitle("%",2);
//    EffGlobD1far->setAxisTitle("%",2);
//    EffGlobD2far->setAxisTitle("%",2);
//    EffGlobD3far->setAxisTitle("%",2);

//     BXGlobDm3->setAxisTitle("%",2);
//     BXGlobDm2->setAxisTitle("%",2);
//     BXGlobDm1->setAxisTitle("%",2);
//     BXGlobD1->setAxisTitle("%",2);
//     BXGlobD2->setAxisTitle("%",2);
//     BXGlobD3->setAxisTitle("%",2);
  
//     BXGlobDm3far->setAxisTitle("%",2);
//     BXGlobDm2far->setAxisTitle("%",2);
//     BXGlobDm1far->setAxisTitle("%",2);
//     BXGlobD1far->setAxisTitle("%",2);
//     BXGlobD2far->setAxisTitle("%",2);
//     BXGlobD3far->setAxisTitle("%",2);

//    MaskedGlobDm3->setAxisTitle("%",2);
//    MaskedGlobDm2->setAxisTitle("%",2);
//    MaskedGlobDm1->setAxisTitle("%",2);
//    MaskedGlobD1->setAxisTitle("%",2);
//    MaskedGlobD2->setAxisTitle("%",2);
//    MaskedGlobD3->setAxisTitle("%",2);
  
//    MaskedGlobDm3far->setAxisTitle("%",2);
//    MaskedGlobDm2far->setAxisTitle("%",2);
//    MaskedGlobDm1far->setAxisTitle("%",2);
//    MaskedGlobD1far->setAxisTitle("%",2);
//    MaskedGlobD2far->setAxisTitle("%",2);
//    MaskedGlobD3far->setAxisTitle("%",2);

//    AverageEffDm3->setAxisTitle("%",2);
//    AverageEffDm2->setAxisTitle("%",2);
//    AverageEffDm1->setAxisTitle("%",2);
//    AverageEffD1->setAxisTitle("%",2);
//    AverageEffD2->setAxisTitle("%",2);
//    AverageEffD3->setAxisTitle("%",2);

//    AverageEffDm3far->setAxisTitle("%",2);
//    AverageEffDm2far->setAxisTitle("%",2);
//    AverageEffDm1far->setAxisTitle("%",2);
//    AverageEffD1far->setAxisTitle("%",2);
//    AverageEffD2far->setAxisTitle("%",2);
//    AverageEffD3far->setAxisTitle("%",2);

//    NoPredictionDm3->setAxisTitle("%",2);
//    NoPredictionDm2->setAxisTitle("%",2);
//    NoPredictionDm1->setAxisTitle("%",2);
//    NoPredictionD1->setAxisTitle("%",2);
//    NoPredictionD2->setAxisTitle("%",2);
//    NoPredictionD3->setAxisTitle("%",2);
  
//    NoPredictionDm3far->setAxisTitle("%",2);
//    NoPredictionDm2far->setAxisTitle("%",2);
//    NoPredictionDm1far->setAxisTitle("%",2);
//    NoPredictionD1far->setAxisTitle("%",2);
//    NoPredictionD2far->setAxisTitle("%",2);
//    NoPredictionD3far->setAxisTitle("%",2);
  }
  

  if(SaveFile){
    std::cout<<"Saving RootFile"<<std::endl;
    dbe->save(NameFile);
  }
  
  if(debug) std::cout<<"RPCEFFICIENCY SECOND DONE"<<std::endl;
  
}

void RPCEfficiencySecond::endJob(){}

