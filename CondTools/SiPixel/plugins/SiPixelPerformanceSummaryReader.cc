#include "SiPixelPerformanceSummaryReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cstdio>
#include <iostream>
#include <sys/time.h>

using namespace cms;
using namespace std;

SiPixelPerformanceSummaryReader::SiPixelPerformanceSummaryReader(const edm::ParameterSet& iConfig)
    : perfSummaryToken_(esConsumes()), printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)) {}

SiPixelPerformanceSummaryReader::~SiPixelPerformanceSummaryReader() = default;

void SiPixelPerformanceSummaryReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::LogInfo("SiPixelPerformanceSummaryReader") << "start reading SiPixelPerformanceSummary" << endl;
  const SiPixelPerformanceSummary* SiPixelPerformanceSummary_ = &iSetup.getData(perfSummaryToken_);
  edm::LogInfo("SiPixelPerformanceSummaryReader") << "end reading SiPixelPerformanceSummary" << endl;

  SiPixelPerformanceSummary_->print();
  vector<uint32_t> allDetIds = SiPixelPerformanceSummary_->getAllDetIds();
  if (!allDetIds.empty())
    SiPixelPerformanceSummary_->print(allDetIds[0]);
  SiPixelPerformanceSummary_->printAll();
}
DEFINE_FWK_MODULE(SiPixelPerformanceSummaryReader);
