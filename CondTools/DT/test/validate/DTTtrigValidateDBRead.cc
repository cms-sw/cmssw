
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

#include "CondTools/DT/test/validate/DTTtrigValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

DTTtrigValidateDBRead::DTTtrigValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTTtrigValidateDBRead::DTTtrigValidateDBRead(int i) {}

DTTtrigValidateDBRead::~DTTtrigValidateDBRead() {}

void DTTtrigValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTTtrig> tT;
  context.get<DTTtrigRcd>().get(tT);
  std::cout << tT->version() << std::endl;
  std::cout << std::distance(tT->begin(), tT->end()) << " data in the container" << std::endl;
  int status;
  int whe;
  int sta;
  int sec;
  int qua;
  float tTrig;
  float tTrms;
  float kFact;
  float cktrig;
  float ckrms;
  float ckfact;
  DTTtrig::const_iterator iter = tT->begin();
  DTTtrig::const_iterator iend = tT->end();
  while (iter != iend) {
    const DTTtrigId& tTId = iter->first;
    const DTTtrigData& tTData = iter->second;
    status = tT->get(tTId.wheelId, tTId.stationId, tTId.sectorId, tTId.slId, tTrig, tTrms, kFact, DTTimeUnits::counts);
    if (status)
      logFile << "ERROR while getting sl Ttrig " << tTId.wheelId << " " << tTId.stationId << " " << tTId.sectorId << " "
              << tTId.slId << " , status = " << status << std::endl;
    if ((fabs(tTData.tTrig - tTrig) > 0.1) || (fabs(tTData.tTrms - tTrms) > 0.0001) ||
        (fabs(tTData.kFact - kFact) > 0.0001))
      logFile << "MISMATCH WHEN READING sl Ttrig " << tTId.wheelId << " " << tTId.stationId << " " << tTId.sectorId
              << " " << tTId.slId << " : " << tTData.tTrig << " " << tTData.tTrms << " " << tTData.kFact << " -> "
              << tTrig << " " << tTrms << " " << kFact << std::endl;
    iter++;
  }

  while (chkFile >> whe >> sta >> sec >> qua >> cktrig >> ckrms >> ckfact) {
    status = tT->get(whe, sta, sec, qua, tTrig, tTrms, kFact, DTTimeUnits::counts);
    if ((fabs(cktrig - tTrig) > 0.1) || (fabs(ckrms - tTrms) > 0.0001) || (fabs(ckfact - kFact) > 0.0001))
      logFile << "MISMATCH IN WRITING AND READING sl Ttrig " << whe << " " << sta << " " << sec << " " << qua << " : "
              << cktrig << " " << ckrms << " " << ckfact << " -> " << tTrig << " " << tTrms << " " << kFact
              << std::endl;
  }
}

void DTTtrigValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "Ttrig validation result:" << std::endl;
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

DEFINE_FWK_MODULE(DTTtrigValidateDBRead);
