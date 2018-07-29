#include "CondTools/Ecal/interface/EcalTPGFineGrainTowerfromFile.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include<iostream>
#include<fstream>
#include <sstream>

popcon::EcalTPGFineGrainTowerfromFile::EcalTPGFineGrainTowerfromFile(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGFineGrainTowerfromFile")) {
  fname = ps.getParameter<std::string>("FileName");

  std::cout << "EcalTPGFineGrainTowerfromFile constructor\n" << std::endl;
}

popcon::EcalTPGFineGrainTowerfromFile::~EcalTPGFineGrainTowerfromFile(){}

void popcon::EcalTPGFineGrainTowerfromFile::getNewObjects() {
  std::cout << "------- Ecal -> getNewObjects\n";
  edm::LogInfo("EcalTPGFineGrainTowerfromFile") << "Started GetNewObjects!!!";
  
  int fileIOV;
  std::cout << "LinPed which input IOV do you want " << std::endl;
  std::cin >> fileIOV; 
  std::ifstream fLin;
  std::ostringstream oss;
  oss << fileIOV;
  //  std::string fname = "/afs/cern.ch/cms/ECAL/triggerTransp/TPG_beamv6_trans_" + oss.str() + "_spikekill.txt";
  fLin.open(fname.c_str());
  if(!fLin.is_open()) {
    std::cout << "ERROR : can't open file '" << fname << std::endl;
    return;
  }
  std::cout << " file " << fname << " opened" << std::endl;
  /*      structure of the file:
TOWER_EB                     365224/375015 : 9792 lines  2448 towers 3 lines : 0, 0, 96 (LUTGroupId, FgGroupId, spike_killing_threshold)
empty line                   375016
TOWER_EE                     375017/379768 : 4752 lines  1584 towers 2 lines : 0, 0x0 (LUTGroupId, tower_lut_fg)
  */
  std::string line;
  for(int i = 0; i < 375016; i++) getline (fLin, line);
  char tower[8];
  unsigned int towerId, LUTFg;
  EcalTPGFineGrainTowerEE * fgrMap = new EcalTPGFineGrainTowerEE;
  for (int itower = 0; itower < 1584; itower++) {
    getline (fLin, line);
    sscanf(line.c_str(), "%s %u", tower, &towerId);
    if(itower < 10 || (itower > 1574 && itower < 1584)) std::cout << " string " << tower << " Id " << towerId;
    getline (fLin, line);    // LUTGroupId
    getline (fLin, line);    // tower_lut_fg
    if(itower < 10 || (itower > 1574 && itower < 1584)) std::cout << " line " << line << std::endl;
    sscanf(line.c_str(), "%x", &LUTFg);
    //    EcalTPGFineGrainTowerEE::Item item;
    //    item.lut = LUTFg;
    //    fgrMap->setValue(towerId, item);

    fgrMap->setValue(towerId, LUTFg);
  }   // end loop over EE towers
  fLin.close();

  m_to_transfer.push_back(std::make_pair(fgrMap, fileIOV));

  std::cout << "Ecal -> end of getNewObjects -----------\n";	
}
