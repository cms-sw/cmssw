
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/DT/test/validate/DTTPGParametersValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"
#include "CondFormats/DataRecord/interface/DTTPGParametersRcd.h"

DTTPGParametersValidateDBRead::DTTPGParametersValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTTPGParametersValidateDBRead::DTTPGParametersValidateDBRead(int i) {}

DTTPGParametersValidateDBRead::~DTTPGParametersValidateDBRead() {}

void DTTPGParametersValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTTPGParameters> tpg;
  context.get<DTTPGParametersRcd>().get(tpg);
  std::cout << tpg->version() << std::endl;
  std::cout << std::distance(tpg->begin(), tpg->end()) << " data in the container" << std::endl;
  int whe;
  int sta;
  int sec;
  int nClock;
  float tPhase;
  int ckclock;
  float ckphase;
  int status;
  DTTPGParameters::const_iterator iter = tpg->begin();
  DTTPGParameters::const_iterator iend = tpg->end();
  while (iter != iend) {
    const DTTPGParametersId& tpgId = iter->first;
    const DTTPGParametersData& tpgData = iter->second;
    status = tpg->get(tpgId.wheelId, tpgId.stationId, tpgId.sectorId, nClock, tPhase, DTTimeUnits::counts);
    if (status)
      logFile << "ERROR while getting cell TPGParameters " << tpgId.wheelId << " " << tpgId.stationId << " "
              << tpgId.sectorId << " , status = " << status << std::endl;
    if ((tpgData.nClock != nClock) || (std::abs(tpgData.tPhase - tPhase) > 0.0001))
      logFile << "MISMATCH WHEN READING cell TPGParameters " << tpgId.wheelId << " " << tpgId.stationId << " "
              << tpgId.sectorId << " : " << tpgData.nClock << " " << tpgData.tPhase << " -> " << nClock << " " << tPhase
              << std::endl;
    iter++;
  }

  while (chkFile >> whe >> sta >> sec >> ckclock >> ckphase) {
    status = tpg->get(whe, sta, sec, nClock, tPhase, DTTimeUnits::counts);
    if ((std::abs(ckclock - nClock) > 0.0001) || (std::abs(ckphase - tPhase) > 0.0001))
      logFile << "MISMATCH IN WRITING AND READING cell TPGParameters " << whe << " " << sta << " " << sec << " : "
              << ckclock << " " << ckphase << " -> " << nClock << " " << tPhase << std::endl;
  }
}

void DTTPGParametersValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "TPGParameters validation result:" << std::endl;
  while (logFile.getline(line, 1000)) {
    std::cout << line << std::endl;
    errors++;
  }
  if (!errors) {
    std::cout << " ********************************* " << std::endl;
    std::cout << " ***                           *** " << std::endl;
    std::cout << " ***      NO ERRORS FOUND      *** " << std::endl;
    std::cout << " ***                           *** " << std::endl;
    std::cout << " ********************************* " << std::endl;
  }
  return;
}

DEFINE_FWK_MODULE(DTTPGParametersValidateDBRead);
