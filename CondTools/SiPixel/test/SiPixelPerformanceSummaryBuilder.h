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
  	    ~SiPixelPerformanceSummaryBuilder() override;
  private:
    void beginJob() override ;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endJob() override ;

  private:
    std::vector<uint32_t> detectorModules_;
  };
}

#endif
