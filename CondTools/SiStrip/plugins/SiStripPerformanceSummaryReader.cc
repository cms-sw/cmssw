#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiStripPerformanceSummaryRcd.h"

#include "CondTools/SiStrip/plugins/SiStripPerformanceSummaryReader.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>

SiStripPerformanceSummaryReader::SiStripPerformanceSummaryReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<int32_t>("printDebug",1)){}

SiStripPerformanceSummaryReader::~SiStripPerformanceSummaryReader(){}

void SiStripPerformanceSummaryReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){
  edm::LogInfo("SiStripPerformanceSummaryReader") << "[SiStripPerformanceSummaryReader::analyze] Start Reading SiStripPerformanceSummary" << std::endl;
  edm::ESHandle<SiStripPerformanceSummary> SiStripPerformanceSummary_;
  iSetup.get<SiStripPerformanceSummaryRcd>().get(SiStripPerformanceSummary_);
  edm::LogInfo("SiStripPerformanceSummaryReader") << "[SiStripPerformanceSummaryReader::analyze] End Reading SiStripPerformanceSummary" << std::endl;
  SiStripPerformanceSummary_->print();
  std::vector<uint32_t> all_detids;
  all_detids.clear();
  SiStripPerformanceSummary_->getDetIds(all_detids);
  if( all_detids.size()>0 ) SiStripPerformanceSummary_->print(all_detids[0]);
  SiStripPerformanceSummary_->printall(); // print all summaries
}
