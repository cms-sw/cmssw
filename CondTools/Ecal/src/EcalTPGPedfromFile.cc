#include "CondTools/Ecal/interface/EcalTPGPedfromFile.h"
#include "CondTools/Ecal/interface/EcalTPGPedestalsHandler.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/LMFSextuple.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

#include<iostream>
#include<iomanip>
#include <sstream>

popcon::EcalTPGPedfromFile::EcalTPGPedfromFile(const edm::ParameterSet & ps) 
  : m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGPedfromFileHandler")) {
  std::cout << "EcalTPGPedfromFile  constructor\n" << std::endl;
}

popcon::EcalTPGPedfromFile::~EcalTPGPedfromFile() {
  // do nothing
}

void popcon::EcalTPGPedfromFile::getNewObjects() {
  std::cout << "------- Ecal -> getNewObjects\n";
  
  unsigned long long max_since = 1;
  Ref payload= lastPayload();
  
  // here popcon tells us which is the last since of the last object in the offline DB
  max_since=tagInfo().lastInterval.first;
  Tm max_since_tm(max_since);

  int fileIOV;
  std::cout << "PedfromFile which input IOV do you want " << std::endl;
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
  for(int i = 0; i < 85; i++) getline (fLin, line);
  char cryst[10];
  uint32_t  ped[kGains], mult[kGains], shift[kGains];
  uint32_t id;
  EcalTPGLinearizationConst *linC = new EcalTPGLinearizationConst;
  EcalTPGPedestals* peds = new EcalTPGPedestals;
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    getline (fLin, line);
    sscanf(line.c_str(), "%s %u", cryst, &id);
    for (int gain = 0; gain < kGains; gain++) {
      getline (fLin, line);
      sscanf(line.c_str(), "%X %X %X", &ped[gain], &mult[gain], &shift[gain]);
    }
    EcalTPGLinearizationConst::Item item;
    item.mult_x1   = mult[2];
    item.mult_x6   = mult[1];
    item.mult_x12  = mult[0];
    item.shift_x1  = shift[2];
    item.shift_x6  = shift[1];
    item.shift_x12 = shift[0];

    EcalTPGPedestals::Item itemPed;
    itemPed.mean_x1  = ped[2];
    itemPed.mean_x6  = ped[1];
    itemPed.mean_x12 = ped[0];

    linC->insert(std::make_pair(id, item));
    peds->insert(std::make_pair(id, itemPed));
  }   // end loop over EB channels
  getline (fLin, line);  // comment before EE crystals
  std::cout << " comment line " << line << std::endl;
  for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
    getline (fLin, line);
    //      std::cout << " line " << line << std::endl;
    sscanf(line.c_str(), "%s %u", cryst, &id);
    //      std::cout << cryst << " id " << id << std::endl; 
    for (int gain =0; gain < kGains; gain++) {
      getline (fLin, line);
      //	std::cout << " line g " << line << std::endl;
      sscanf(line.c_str(), "%X %X %X", &ped[gain], &mult[gain], &shift[gain]);
      //	std::cout << " gain " << gain << " ped " << ped[gain] << " mult " << mult[gain] << " shift " << shift[gain]<< std::endl;

    }
    EcalTPGLinearizationConst::Item item;
    item.mult_x1   = mult[2];
    item.mult_x6   = mult[1];
    item.mult_x12  = mult[0];
    item.shift_x1  = shift[2];
    item.shift_x6  = shift[1];
    item.shift_x12 = shift[0];

    EcalTPGPedestals::Item itemPed;
    itemPed.mean_x1  = ped[2];
    itemPed.mean_x6  = ped[1];
    itemPed.mean_x12 = ped[0];

    linC->insert(std::make_pair(id, item));
    peds->insert(std::make_pair(id, itemPed));
  }   // end loop over EE channels
  fLin.close();
  // for the time beeing just transfer pedestal
  //  m_to_transfer.push_back(std::make_pair(linC, fileIOV));
  m_to_transfer.push_back(std::make_pair(peds, fileIOV));

  std::cout << "Ecal -> end of getNewObjects -----------\n";	
}
