#ifndef SiPixelPerformanceSummaryReader_H
#define SiPixelPerformanceSummaryReader_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace cms {
  class SiPixelPerformanceSummaryReader : public edm::EDAnalyzer {
  public:
    explicit SiPixelPerformanceSummaryReader(const edm::ParameterSet&);
            ~SiPixelPerformanceSummaryReader();
  
    void analyze(const edm::Event&, const edm::EventSetup&);

  private:
    bool printdebug_;
  };
}

#endif
