#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"

#include "CondTools/SiPixel/test/SiPixelPerformanceSummaryReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>


using namespace cms;


SiPixelPerformanceSummaryReader::SiPixelPerformanceSummaryReader(const edm::ParameterSet& iConfig) 
                               : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)) {}


SiPixelPerformanceSummaryReader::~SiPixelPerformanceSummaryReader() {}


void SiPixelPerformanceSummaryReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::LogInfo("SiPixelPerformanceSummaryReader") << "start reading SiPixelPerformanceSummary" << std::endl;
  edm::ESHandle<SiPixelPerformanceSummary> SiPixelPerformanceSummary_;
  iSetup.get<SiPixelPerformanceSummaryRcd>().get(SiPixelPerformanceSummary_);
  edm::LogInfo("SiPixelPerformanceSummaryReader") << "end reading SiPixelPerformanceSummary" << std::endl;

  SiPixelPerformanceSummary_->print();
  std::vector<uint32_t> allDetIds;
                        allDetIds.clear();
  SiPixelPerformanceSummary_->getAllDetIds(allDetIds);
  if (allDetIds.size()>0) SiPixelPerformanceSummary_->print(allDetIds[0]);
  //SiPixelPerformanceSummary_->printAll(); 
}
