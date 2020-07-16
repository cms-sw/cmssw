
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

#include "CondTools/DT/test/validate/DTDeadFlagValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"

DTDeadFlagValidateDBRead::DTDeadFlagValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTDeadFlagValidateDBRead::DTDeadFlagValidateDBRead(int i) {}

DTDeadFlagValidateDBRead::~DTDeadFlagValidateDBRead() {}

void DTDeadFlagValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTDeadFlag> df;
  context.get<DTDeadFlagRcd>().get(df);
  std::cout << df->version() << std::endl;
  std::cout << std::distance(df->begin(), df->end()) << " data in the container" << std::endl;
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  bool ckdead_HV;
  bool ckdead_TP;
  bool ckdead_RO;
  bool ckdiscCat;

  int status;
  DTDeadFlag::const_iterator iter = df->begin();
  DTDeadFlag::const_iterator iend = df->end();
  while (iter != iend) {
    const DTDeadFlagId& dfId = iter->first;
    const DTDeadFlagData& dfData = iter->second;
    status = df->get(dfId.wheelId,
                     dfId.stationId,
                     dfId.sectorId,
                     dfId.slId,
                     dfId.layerId,
                     dfId.cellId,
                     dead_HV,
                     dead_TP,
                     dead_RO,
                     discCat);
    if (status)
      logFile << "ERROR while getting cell flags " << dfId.wheelId << " " << dfId.stationId << " " << dfId.sectorId
              << " " << dfId.slId << " " << dfId.layerId << " " << dfId.cellId << " , status = " << status << std::endl;
    if ((dfData.dead_HV ^ dead_HV) || (dfData.dead_TP ^ dead_TP) || (dfData.dead_RO ^ dead_RO) ||
        (dfData.discCat ^ discCat))
      logFile << "MISMATCH WHEN READING cell flags " << dfId.wheelId << " " << dfId.stationId << " " << dfId.sectorId
              << " " << dfId.slId << " " << dfId.layerId << " " << dfId.cellId << " : " << dfData.dead_HV << " "
              << dfData.dead_TP << " " << dfData.dead_RO << " " << dfData.discCat << " -> " << dead_HV << " " << dead_TP
              << " " << dead_RO << " " << discCat << std::endl;
    iter++;
  }

  while (chkFile >> whe >> sta >> sec >> qua >> lay >> cel >> ckdead_HV >> ckdead_TP >> ckdead_RO >> ckdiscCat) {
    status = df->get(whe, sta, sec, qua, lay, cel, dead_HV, dead_TP, dead_RO, discCat);
    if ((ckdead_HV ^ dead_HV) || (ckdead_TP ^ dead_TP) || (ckdead_RO ^ dead_RO) || (ckdiscCat ^ discCat))
      logFile << "MISMATCH IN WRITING AND READING cell flags " << whe << " " << sta << " " << sec << " " << qua << " "
              << lay << " " << cel << " : " << ckdead_HV << " " << ckdead_TP << " " << ckdead_RO << " " << ckdiscCat
              << " -> " << dead_HV << " " << dead_TP << " " << dead_RO << " " << discCat << std::endl;
  }
}

void DTDeadFlagValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "DeadFlags validation result:" << std::endl;
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

DEFINE_FWK_MODULE(DTDeadFlagValidateDBRead);
