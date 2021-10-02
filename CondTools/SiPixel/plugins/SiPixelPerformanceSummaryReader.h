#ifndef SiPixelPerformanceSummaryReader_H
#define SiPixelPerformanceSummaryReader_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiPixelPerformanceSummaryRcd.h"

namespace cms {
  class SiPixelPerformanceSummaryReader : public edm::one::EDAnalyzer<> {
  public:
    explicit SiPixelPerformanceSummaryReader(const edm::ParameterSet&);
    ~SiPixelPerformanceSummaryReader() override;

    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    const edm::ESGetToken<SiPixelPerformanceSummary, SiPixelPerformanceSummaryRcd> perfSummaryToken_;
    bool printdebug_;
  };
}  // namespace cms

#endif
