#include "CondTools/Ecal/interface/EcalTPGPhysicsConstfromFile.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>
#include<fstream>
#include <sstream>

popcon::EcalTPGPhysicsConstfromFile::EcalTPGPhysicsConstfromFile(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGPhysicsConstfromFile")) {

  std::cout << "EcalTPGPhysicsConstfromFile constructor" << std::endl;
}

popcon::EcalTPGPhysicsConstfromFile::~EcalTPGPhysicsConstfromFile() {
  // do nothing
}

void popcon::EcalTPGPhysicsConstfromFile::getNewObjects() {
  std::cout << "------- EcalTPGPhysicsConstfromFile -> getNewObjects\n";
	edm::LogInfo("EcalTPGPhysicsConstfromFile") << "Started GetNewObjects!!!";

  Ref payload= lastPayload();  
  // here popcon tells us which is the last since of the last object in the offline DB

  int fileIOV;
  std::cout << "LinPed which input IOV do you want " << std::endl;
  std::cin >> fileIOV; 
  std::ifstream fLin;
  std::ostringstream oss;
  oss << fileIOV;
  std::string fname = "/afs/cern.ch/cms/ECAL/triggerTransp/TPG_beamv6_trans_" + oss.str() + "_spikekill.txt";
  fLin.open(fname.c_str());
  if(!fLin.is_open()) {
    std::cout << "ERROR : can't open file '" << fname << std::endl;
    return;
  }
  std::cout << " file " << fname << " opened" << std::endl;
  std::string line;
  for(int i = 0; i < 76; i++) getline (fLin, line);

  EcalTPGPhysicsConst::Item item;
  // Ecal barrel detector	  	              
  getline (fLin, line);   // PHYSICS_EB 838860800
  //  std::cout << " EB DetId " << line << std::endl;
  DetId eb(DetId::Ecal, EcalBarrel);
  float ETSat, TTThreshlow, TTThreshhigh, FG_lowThreshold, FG_highThreshold, FG_lowRatio, FG_highRatio;
  getline (fLin, line);
  sscanf(line.c_str(), "%f %f %f", &ETSat, &TTThreshlow, &TTThreshhigh);
  item.EtSat = ETSat;
  item.ttf_threshold_Low = TTThreshlow;
  item.ttf_threshold_High = TTThreshhigh;
  getline (fLin, line);
  sscanf(line.c_str(), "%f %f %f %f", &FG_lowThreshold, &FG_highThreshold, &FG_lowRatio, &FG_highRatio);
  item.FG_lowThreshold = FG_lowThreshold; 
  item.FG_highThreshold = FG_highThreshold; 
  item.FG_lowRatio = FG_lowRatio; 
  item.FG_highRatio = FG_highRatio; 
  EcalTPGPhysicsConst* physC = new EcalTPGPhysicsConst;
  physC->setValue(eb.rawId(), item);

  // Ecal endcap detector	  	              
  getline (fLin, line);   // empty line
  getline (fLin, line);   // PHYSICS_EE 872415232
  std::cout << " EE DetId " << line << std::endl;
  DetId ee(DetId::Ecal, EcalEndcap);
  getline (fLin, line);
  //  std::cout << " EE TTT " << line << std::endl;
  sscanf(line.c_str(), "%f %f %f", &ETSat, &TTThreshlow, &TTThreshhigh);
  item.EtSat = ETSat;
  item.ttf_threshold_Low = TTThreshlow;
  item.ttf_threshold_High = TTThreshhigh;
  getline (fLin, line);
  //  std::cout << " EE FG " << line << std::endl;
  sscanf(line.c_str(), "%f %f %f %f", &FG_lowThreshold, &FG_highThreshold, &FG_lowRatio, &FG_highRatio);
  item.FG_lowThreshold = FG_lowThreshold; 
  item.FG_highThreshold = FG_highThreshold; 
  item.FG_lowRatio = FG_lowRatio; 
  item.FG_highRatio = FG_highRatio; 
  physC->setValue(ee.rawId(), item); 

  m_to_transfer.push_back(std::make_pair(physC, fileIOV));

  std::cout << "EcalTPGPhysicsConstfromFile - > end of getNewObjects -----------\n";
	
}
