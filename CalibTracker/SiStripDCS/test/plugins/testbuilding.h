#ifndef CalibTracker_SiStripDCS_test_testbuilding_H
#define CalibTracker_SiStripDCS_test_testbuilding_H

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVBuilder.h"

class testbuilding : public edm::EDAnalyzer {
 public:
  testbuilding(const edm::ParameterSet&);
  ~testbuilding();
  
 private:
  void beginRun(const edm::Run&, const edm::EventSetup& );
  void analyze( const edm::Event&, const edm::EventSetup& ) {;}

  //  SiStripModuleHVBuilder *hvBuilder;
  edm::Service<SiStripModuleHVBuilder> hvBuilder;
};
#endif

