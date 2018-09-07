#include "CondTools/Ecal/interface/EcalTPGFineGrainStripfromFile.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include<iostream>
#include<fstream>
#include <sstream>

popcon::EcalTPGFineGrainStripfromFile::EcalTPGFineGrainStripfromFile(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGFineGrainStripfromFile")) {
  fname = ps.getParameter<std::string>("FileName");

  std::cout << "EcalTPGFineGrainStripfromFile constructor\n" << std::endl;
}

popcon::EcalTPGFineGrainStripfromFile::~EcalTPGFineGrainStripfromFile(){
  // do nothing
}

void popcon::EcalTPGFineGrainStripfromFile::getNewObjects() {
  std::cout << "------- Ecal -> getNewObjects\n";
	edm::LogInfo("EcalTPGFineGrainStripfromFile") << "Started GetNewObjects!!!";
  
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
  std::string line;
  for(int i = 0; i < 304517; i++) getline (fLin, line);
  char strip[8];
  unsigned int stripId, Threshold, LUTFgr;

  EcalTPGFineGrainStripEE * fgrStripEE = new EcalTPGFineGrainStripEE;

  for (int istrip = 0; istrip < 15176; istrip++) {
    getline (fLin, line);
    sscanf(line.c_str(), "%s %u", strip, &stripId);
    if(istrip < 10 || (istrip > 12239 && istrip < 12250)) std::cout << " string " << strip << " Id " << stripId;
    getline (fLin, line);    // sliding_window
    getline (fLin, line);    // weightGroupId
    getline (fLin, line);    // threshold_sfg lut_sfg
    if(istrip < 10 || (istrip > 12239 && istrip < 12250)) std::cout << " line " << line;
    sscanf(line.c_str(), "%x %x", &Threshold, &LUTFgr);
    if(istrip < 10 || (istrip > 12239 && istrip < 12250)) std::cout  << " Threshold " << Threshold << std::endl;
    EcalTPGFineGrainStripEE::Item item;
    item.threshold = Threshold;
    item.lut = LUTFgr;

    fgrStripEE->setValue(stripId, item);
    if(istrip == 12239) getline (fLin, line);    // 1 empty line between EB and EE
  }   // end loop over EB + EE strips
  fLin.close();

  m_to_transfer.push_back(std::make_pair(fgrStripEE, fileIOV));

  std::cout << "Ecal -> end of getNewObjects -----------\n";	
}
