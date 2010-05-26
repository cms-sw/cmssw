#ifndef SiStripPerformanceSummaryReader_H
#define SiStripPerformanceSummaryReader_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripPerformanceSummaryReader : public edm::EDAnalyzer {

 public:
  explicit SiStripPerformanceSummaryReader( const edm::ParameterSet& );
  ~SiStripPerformanceSummaryReader();
  
  void analyze( const edm::Event&, const edm::EventSetup& );
  
 private:
  uint32_t printdebug_;
};
#endif
