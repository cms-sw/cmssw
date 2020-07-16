
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

#include "CondTools/DT/test/validate/DTLVStatusValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTLVStatus.h"
#include "CondFormats/DataRecord/interface/DTLVStatusRcd.h"

DTLVStatusValidateDBRead::DTLVStatusValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTLVStatusValidateDBRead::DTLVStatusValidateDBRead(int i) {}

DTLVStatusValidateDBRead::~DTLVStatusValidateDBRead() {}

void DTLVStatusValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTLVStatus> lv;
  context.get<DTLVStatusRcd>().get(lv);
  std::cout << lv->version() << std::endl;
  std::cout << std::distance(lv->begin(), lv->end()) << " data in the container" << std::endl;
  int status;
  int whe;
  int sta;
  int sec;
  int flagCFE;
  int flagDFE;
  int flagCMC;
  int flagDMC;
  int ckflagCFE;
  int ckflagDFE;
  int ckflagCMC;
  int ckflagDMC;

  DTLVStatus::const_iterator iter = lv->begin();
  DTLVStatus::const_iterator iend = lv->end();
  while (iter != iend) {
    const DTLVStatusId& lvId = iter->first;
    const DTLVStatusData& lvData = iter->second;
    status = lv->get(lvId.wheelId, lvId.stationId, lvId.sectorId, flagCFE, flagDFE, flagCMC, flagDMC);
    if (status)
      logFile << "ERROR while getting LV status" << lvId.wheelId << " " << lvId.stationId << " " << lvId.sectorId
              << " , status = " << status << std::endl;
    if ((lvData.flagCFE != flagCFE) || (lvData.flagDFE != flagDFE) || (lvData.flagCMC != flagCMC) ||
        (lvData.flagDMC != flagDMC))
      logFile << "MISMATCH WHEN READING LV status" << lvId.wheelId << " " << lvId.stationId << " " << lvId.sectorId
              << " : " << lvData.flagCFE << " " << lvData.flagDFE << " " << lvData.flagCMC << " " << lvData.flagDMC
              << " -> " << flagCFE << " " << flagDFE << " " << flagCMC << " " << flagDMC << std::endl;
    iter++;
  }

  while (chkFile >> whe >> sta >> sec >> ckflagCFE >> ckflagDFE >> ckflagCMC >> ckflagDMC) {
    status = lv->get(whe, sta, sec, flagCFE, flagDFE, flagCMC, flagDMC);
    if ((ckflagCFE != flagCFE) || (ckflagDFE != flagDFE) || (ckflagCMC != flagCMC) || (ckflagDMC != flagDMC))
      logFile << "MISMATCH IN WRITING AND READING LV status " << whe << " " << sta << " " << sec << " : " << ckflagCFE
              << " " << ckflagDFE << " " << ckflagCMC << " " << ckflagDMC << " -> " << flagCFE << " " << flagDFE << " "
              << flagCMC << " " << flagDMC << std::endl;
  }
}

void DTLVStatusValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "LV status validation result:" << std::endl;
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

DEFINE_FWK_MODULE(DTLVStatusValidateDBRead);
