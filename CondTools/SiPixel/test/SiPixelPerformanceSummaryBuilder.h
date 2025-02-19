#ifndef SiPixelPerformanceSummaryBuilder_H
#define SiPixelPerformanceSummaryBuilder_H


#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace cms {
  class SiPixelPerformanceSummaryBuilder : public edm::EDAnalyzer {
  public:
    explicit SiPixelPerformanceSummaryBuilder(const edm::ParameterSet&);
  	    ~SiPixelPerformanceSummaryBuilder();
  private:
    virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;

  private:
    std::vector<uint32_t> detectorModules_;
  };
}

#endif
