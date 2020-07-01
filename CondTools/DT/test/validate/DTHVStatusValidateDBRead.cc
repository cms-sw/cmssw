
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

#include "CondTools/DT/test/validate/DTHVStatusValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "CondFormats/DataRecord/interface/DTHVStatusRcd.h"

DTHVStatusValidateDBRead::DTHVStatusValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTHVStatusValidateDBRead::DTHVStatusValidateDBRead(int i) {}

DTHVStatusValidateDBRead::~DTHVStatusValidateDBRead() {}

void DTHVStatusValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTHVStatus> hv;
  context.get<DTHVStatusRcd>().get(hv);
  std::cout << hv->version() << std::endl;
  std::cout << std::distance(hv->begin(), hv->end()) << " data in the container" << std::endl;
  int status;
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int l_p;
  int fCell;
  int lCell;
  int flagA;
  int flagC;
  int flagS;
  int ckfCell;
  int cklCell;
  int ckflagA;
  int ckflagC;
  int ckflagS;

  DTHVStatus::const_iterator iter = hv->begin();
  DTHVStatus::const_iterator iend = hv->end();
  while (iter != iend) {
    const DTHVStatusId& hvId = iter->first;
    const DTHVStatusData& hvData = iter->second;
    status = hv->get(hvId.wheelId,
                     hvId.stationId,
                     hvId.sectorId,
                     hvId.slId,
                     hvId.layerId,
                     hvId.partId,
                     fCell,
                     lCell,
                     flagA,
                     flagC,
                     flagS);
    if (status)
      logFile << "ERROR while getting HV status" << hvId.wheelId << " " << hvId.stationId << " " << hvId.sectorId << " "
              << hvId.slId << " " << hvId.layerId << " " << hvId.partId << " , status = " << status << std::endl;
    if ((hvData.fCell != fCell) || (hvData.lCell != lCell) || (hvData.flagA != flagA) || (hvData.flagC != flagC) ||
        (hvData.flagS != flagS))
      logFile << "MISMATCH WHEN READING HV status" << hvId.wheelId << " " << hvId.stationId << " " << hvId.sectorId
              << " " << hvId.slId << " : " << hvData.fCell << " " << hvData.lCell << " " << hvData.flagA << " "
              << hvData.flagC << " " << hvData.flagS << " -> " << fCell << " " << lCell << " " << flagA << " " << flagC
              << " " << flagS << std::endl;
    iter++;
  }

  while (chkFile >> whe >> sta >> sec >> qua >> lay >> l_p >> ckfCell >> cklCell >> ckflagA >> ckflagC >> ckflagS) {
    status = hv->get(whe, sta, sec, qua, lay, l_p, fCell, lCell, flagA, flagC, flagS);
    if (!flagA && !flagC && !flagS)
      ckfCell = cklCell = 0;
    if ((ckfCell != fCell) || (cklCell != lCell) || (ckflagA != flagA) || (ckflagC != flagC) || (ckflagS != flagS))
      logFile << "MISMATCH IN WRITING AND READING HV status " << whe << " " << sta << " " << sec << " " << qua << " : "
              << ckfCell << " " << cklCell << " " << ckflagA << " " << ckflagC << " " << ckflagS << " -> " << fCell
              << " " << lCell << " " << flagA << " " << flagC << " " << flagS << std::endl;
  }
}

void DTHVStatusValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "HV status validation result:" << std::endl;
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

DEFINE_FWK_MODULE(DTHVStatusValidateDBRead);
