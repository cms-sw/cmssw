#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"

#include "CondTools/SiPixel/test/SiPixelPerformanceSummaryReader.h"

#include <cstdio>
#include <iostream>
#include <sys/time.h>

using namespace cms;
using namespace std;

SiPixelPerformanceSummaryReader::SiPixelPerformanceSummaryReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)) {}

SiPixelPerformanceSummaryReader::~SiPixelPerformanceSummaryReader() {}

void SiPixelPerformanceSummaryReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::LogInfo("SiPixelPerformanceSummaryReader") << "start reading SiPixelPerformanceSummary" << endl;
  edm::ESHandle<SiPixelPerformanceSummary> SiPixelPerformanceSummary_;
  iSetup.get<SiPixelPerformanceSummaryRcd>().get(SiPixelPerformanceSummary_);
  edm::LogInfo("SiPixelPerformanceSummaryReader") << "end reading SiPixelPerformanceSummary" << endl;

  SiPixelPerformanceSummary_->print();
  vector<uint32_t> allDetIds = SiPixelPerformanceSummary_->getAllDetIds();
  if (!allDetIds.empty())
    SiPixelPerformanceSummary_->print(allDetIds[0]);
  SiPixelPerformanceSummary_->printAll();
}
