
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

#include "CondTools/DT/test/validate/DTT0ValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

DTT0ValidateDBRead::DTT0ValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTT0ValidateDBRead::DTT0ValidateDBRead(int i) {}

DTT0ValidateDBRead::~DTT0ValidateDBRead() {}

void DTT0ValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTT0> t0;
  context.get<DTT0Rcd>().get(t0);
  std::cout << t0->version() << std::endl;
  std::cout << std::distance(t0->begin(), t0->end()) << " data in the container" << std::endl;
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;
  float t0mean;
  float t0rms;
  float ckmean;
  float ckrms;
  int status;
  DTT0::const_iterator iter = t0->begin();
  DTT0::const_iterator iend = t0->end();
  while (iter != iend) {
    const DTT0Data& t0Data = *iter++;
    int channelId = t0Data.channelId;
    if (channelId == 0)
      continue;
    DTWireId id(channelId);
    status = t0->get(id, t0mean, t0rms, DTTimeUnits::counts);
    if (status)
      logFile << "ERROR while getting cell T0 " << id.wheel() << " " << id.station() << " " << id.sector() << " "
              << id.superlayer() << " " << id.layer() << " " << id.wire() << " , status = " << status << std::endl;
    if ((fabs(t0Data.t0mean - t0mean) > 0.0001) || (fabs(t0Data.t0rms - t0rms) > 0.0001))
      logFile << "MISMATCH WHEN READING cell T0 " << id.wheel() << " " << id.station() << " " << id.sector() << " "
              << id.superlayer() << " " << id.layer() << " " << id.wire() << " : " << t0Data.t0mean << " "
              << t0Data.t0rms << " -> " << t0mean << " " << t0rms << std::endl;
  }

  while (chkFile >> whe >> sta >> sec >> qua >> lay >> cel >> ckmean >> ckrms) {
    status = t0->get(whe, sta, sec, qua, lay, cel, t0mean, t0rms, DTTimeUnits::counts);
    if ((fabs(ckmean - t0mean) > 0.0001) || (fabs(ckrms - t0rms) > 0.0001))
      logFile << "MISMATCH IN WRITING AND READING cell T0 " << whe << " " << sta << " " << sec << " " << qua << " "
              << lay << " " << cel << " : " << ckmean << " " << ckrms << " -> " << t0mean << " " << t0rms << std::endl;
  }
}

void DTT0ValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "T0 validation result:" << std::endl;
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

DEFINE_FWK_MODULE(DTT0ValidateDBRead);
