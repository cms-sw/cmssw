#include "CondTools/Ecal/interface/EcalTPGSpikeThresholdfromFile.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include<iostream>
#include<fstream>
#include <sstream>

popcon::EcalTPGSpikeThresholdfromFile::EcalTPGSpikeThresholdfromFile(const edm::ParameterSet & ps)
  :    m_name(ps.getUntrackedParameter<std::string>("name","EcalTPGSpikeThresholdfromFile")) {

  std::cout << "EcalTPGSpikeThresholdfromFile constructor\n" << std::endl;
}

popcon::EcalTPGSpikeThresholdfromFile::~EcalTPGSpikeThresholdfromFile(){
  // do nothing
}

void popcon::EcalTPGSpikeThresholdfromFile::getNewObjects() {
  std::cout << "------- Ecal -> getNewObjects\n";
	edm::LogInfo("EcalTPGSpikeThresholdfromFile") << "Started GetNewObjects!!!";
  
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
  for(int i = 0; i < 365223; i++) getline (fLin, line);
  char tow[8];
  unsigned int towerId, Threshold;
  EcalTPGSpike *lut=new EcalTPGSpike();
  for (int istrip = 0; istrip < 2448; istrip++) {
    getline (fLin, line);
    sscanf(line.c_str(), "%s %u", tow, &towerId);
    if(istrip < 10) std::cout << " string " << tow << " Id " << towerId;
    getline (fLin, line);    // LUTGroupId
    getline (fLin, line);    // FgGroupId
    getline (fLin, line);    // spike_killing_threshold
    if(istrip < 10) std::cout << " line " << line;
    sscanf(line.c_str(), "%u", &Threshold);
    if(istrip < 10) std::cout  << " Threshold " << Threshold << std::endl;

    lut->setValue(towerId, Threshold);
  }   // end loop over EB towers
  fLin.close();

  m_to_transfer.push_back(std::make_pair(lut, fileIOV));

  std::cout << "Ecal -> end of getNewObjects -----------\n";	
}
