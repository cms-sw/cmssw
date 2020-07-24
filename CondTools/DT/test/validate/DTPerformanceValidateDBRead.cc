
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

#include "CondTools/DT/test/validate/DTPerformanceValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTPerformance.h"
#include "CondFormats/DataRecord/interface/DTPerformanceRcd.h"

DTPerformanceValidateDBRead::DTPerformanceValidateDBRead(edm::ParameterSet const& p)
    : dataFileName(p.getParameter<std::string>("chkFile")), elogFileName(p.getParameter<std::string>("logFile")) {}

DTPerformanceValidateDBRead::DTPerformanceValidateDBRead(int i) {}

DTPerformanceValidateDBRead::~DTPerformanceValidateDBRead() {}

void DTPerformanceValidateDBRead::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
  std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile(run_fn.str().c_str());
  std::ofstream logFile(elogFileName.c_str(), std::ios_base::app);
  edm::ESHandle<DTPerformance> mP;
  context.get<DTPerformanceRcd>().get(mP);
  std::cout << mP->version() << std::endl;
  std::cout << std::distance(mP->begin(), mP->end()) << " data in the container" << std::endl;
  int status;
  int whe;
  int sta;
  int sec;
  int qua;
  float meanT0;
  float meanTtrig;
  float meanMtime;
  float meanNoise;
  float meanAfterPulse;
  float meanResolution;
  float meanEfficiency;
  float ckmeanT0;
  float ckmeanTtrig;
  float ckmeanMtime;
  float ckmeanNoise;
  float ckmeanAfterPulse;
  float ckmeanResolution;
  float ckmeanEfficiency;

  DTPerformance::const_iterator iter = mP->begin();
  DTPerformance::const_iterator iend = mP->end();
  while (iter != iend) {
    const DTPerformanceId& mpId = iter->first;
    const DTPerformanceData& mpData = iter->second;
    status = mP->get(mpId.wheelId,
                     mpId.stationId,
                     mpId.sectorId,
                     mpId.slId,
                     meanT0,
                     meanTtrig,
                     meanMtime,
                     meanNoise,
                     meanAfterPulse,
                     meanResolution,
                     meanEfficiency,
                     DTTimeUnits::counts);
    if (status)
      logFile << "ERROR while getting sl performance " << mpId.wheelId << " " << mpId.stationId << " " << mpId.sectorId
              << " " << mpId.slId << " , status = " << status << std::endl;
    if ((fabs(mpData.meanT0 - meanT0) > 0.0001) || (fabs(mpData.meanTtrig - meanTtrig) > 0.1) ||
        (fabs(mpData.meanMtime - meanMtime) > 0.0001) || (fabs(mpData.meanNoise - meanNoise) > 0.0001) ||
        (fabs(mpData.meanAfterPulse - meanAfterPulse) > 0.0001) ||
        (fabs(mpData.meanResolution - meanResolution) > 0.0001) ||
        (fabs(mpData.meanEfficiency - meanEfficiency) > 0.0001))
      logFile << "MISMATCH WHEN READING sl performance " << mpId.wheelId << " " << mpId.stationId << " "
              << mpId.sectorId << " " << mpId.slId << " : " << mpData.meanT0 << " " << mpData.meanTtrig << " "
              << mpData.meanMtime << " " << mpData.meanNoise << " " << mpData.meanAfterPulse << " "
              << mpData.meanResolution << " " << mpData.meanEfficiency << " -> " << meanT0 << " " << meanTtrig << " "
              << meanMtime << " " << meanNoise << " " << meanAfterPulse << " " << meanResolution << " "
              << meanEfficiency << std::endl;
    iter++;
  }

  while (chkFile >> whe >> sta >> sec >> qua >> ckmeanT0 >> ckmeanTtrig >> ckmeanMtime >> ckmeanNoise >>
         ckmeanAfterPulse >> ckmeanResolution >> ckmeanEfficiency) {
    status = mP->get(whe,
                     sta,
                     sec,
                     qua,
                     meanT0,
                     meanTtrig,
                     meanMtime,
                     meanNoise,
                     meanAfterPulse,
                     meanResolution,
                     meanEfficiency,
                     DTTimeUnits::counts);
    if ((fabs(ckmeanT0 - meanT0) > 0.0001) || (fabs(ckmeanTtrig - meanTtrig) > 0.1) ||
        (fabs(ckmeanMtime - meanMtime) > 0.01) || (fabs(ckmeanNoise - meanNoise) > 0.0001) ||
        (fabs(ckmeanAfterPulse - meanAfterPulse) > 0.0001) || (fabs(ckmeanResolution - meanResolution) > 0.01) ||
        (fabs(ckmeanEfficiency - meanEfficiency) > 0.0001))
      logFile << "MISMATCH IN WRITING AND READING sl performance " << whe << " " << sta << " " << sec << " " << qua
              << " : " << ckmeanT0 << " " << ckmeanTtrig << " " << ckmeanMtime << " " << ckmeanNoise << " "
              << ckmeanAfterPulse << " " << ckmeanResolution << " " << ckmeanEfficiency << " -> " << meanT0 << " "
              << meanTtrig << " " << meanMtime << " " << meanNoise << " " << meanAfterPulse << " " << meanResolution
              << " " << meanEfficiency << std::endl;
  }
}

void DTPerformanceValidateDBRead::endJob() {
  std::ifstream logFile(elogFileName.c_str());
  char* line = new char[1000];
  int errors = 0;
  std::cout << "Mean performance validation result:" << std::endl;
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

DEFINE_FWK_MODULE(DTPerformanceValidateDBRead);
