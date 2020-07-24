
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

#include "CondTools/DT/test/validate/DTStatusFlagValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"

DTStatusFlagValidateDBRead::DTStatusFlagValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTStatusFlagValidateDBRead::DTStatusFlagValidateDBRead(int i) {}

DTStatusFlagValidateDBRead::~DTStatusFlagValidateDBRead() {}

void DTStatusFlagValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTStatusFlag> sf;
  context.get<DTStatusFlagRcd>().get(sf);
  std::cout << sf->version() << std::endl;
  std::cout << std::distance(sf->begin(), sf->end()) << " data in the container" << std::endl;
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;

  bool noiseFlag;
  bool feMask;
  bool tdcMask;
  bool trigMask;
  bool deadFlag;
  bool nohvFlag;
  bool cknoiseFlag;
  bool ckfeMask;
  bool cktdcMask;
  bool cktrigMask;
  bool ckdeadFlag;
  bool cknohvFlag;

  int status;
  DTStatusFlag::const_iterator iter = sf->begin();
  DTStatusFlag::const_iterator iend = sf->end();
  while (iter != iend) {
    const DTStatusFlagId& sfId = iter->first;
    const DTStatusFlagData& sfData = iter->second;
    status = sf->get(sfId.wheelId,
                     sfId.stationId,
                     sfId.sectorId,
                     sfId.slId,
                     sfId.layerId,
                     sfId.cellId,
                     noiseFlag,
                     feMask,
                     tdcMask,
                     trigMask,
                     deadFlag,
                     nohvFlag);
    if (status)
      logFile << "ERROR while getting cell flags " << sfId.wheelId << " " << sfId.stationId << " " << sfId.sectorId
              << " " << sfId.slId << " " << sfId.layerId << " " << sfId.cellId << " , status = " << status << std::endl;
    if ((sfData.noiseFlag ^ noiseFlag) || (sfData.feMask ^ feMask) || (sfData.tdcMask ^ tdcMask) ||
        (sfData.trigMask ^ trigMask) || (sfData.deadFlag ^ deadFlag) || (sfData.nohvFlag ^ nohvFlag))
      logFile << "MISMATCH WHEN READING cell flags " << sfId.wheelId << " " << sfId.stationId << " " << sfId.sectorId
              << " " << sfId.slId << " " << sfId.layerId << " " << sfId.cellId << " : " << sfData.noiseFlag << " "
              << sfData.feMask << " " << sfData.tdcMask << " " << sfData.trigMask << " " << sfData.deadFlag << " "
              << sfData.nohvFlag << " -> " << noiseFlag << " " << feMask << " " << tdcMask << " " << trigMask << " "
              << deadFlag << " " << nohvFlag << std::endl;
    iter++;
  }

  while (chkFile >> whe >> sta >> sec >> qua >> lay >> cel >> cknoiseFlag >> ckfeMask >> cktdcMask >> cktrigMask >>
         ckdeadFlag >> cknohvFlag) {
    status = sf->get(whe, sta, sec, qua, lay, cel, noiseFlag, feMask, tdcMask, trigMask, deadFlag, nohvFlag);
    if ((cknoiseFlag ^ noiseFlag) || (ckfeMask ^ feMask) || (cktdcMask ^ tdcMask) || (cktrigMask ^ trigMask) ||
        (ckdeadFlag ^ deadFlag) || (cknohvFlag ^ nohvFlag))
      logFile << "MISMATCH IN WRITING AND READING cell flags " << whe << " " << sta << " " << sec << " " << qua << " "
              << lay << " " << cel << " : " << cknoiseFlag << " " << ckfeMask << " " << cktdcMask << " " << cktrigMask
              << " " << ckdeadFlag << " " << cknohvFlag << " -> " << noiseFlag << " " << feMask << " " << tdcMask << " "
              << trigMask << " " << deadFlag << " " << nohvFlag << std::endl;
  }
}

void DTStatusFlagValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "StatusFlags validation result:" << std::endl;
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

DEFINE_FWK_MODULE(DTStatusFlagValidateDBRead);
