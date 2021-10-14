#ifndef SiPixelPerformanceSummaryBuilder_H
#define SiPixelPerformanceSummaryBuilder_H

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace cms {
  class SiPixelPerformanceSummaryBuilder : public edm::one::EDAnalyzer<> {
  public:
    explicit SiPixelPerformanceSummaryBuilder(const edm::ParameterSet&);
    ~SiPixelPerformanceSummaryBuilder() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
    std::vector<uint32_t> detectorModules_;
  };
}  // namespace cms

#endif
