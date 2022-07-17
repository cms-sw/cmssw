#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

class DumpMkFitGeometry : public edm::one::EDAnalyzer<> {
public:
  explicit DumpMkFitGeometry(const edm::ParameterSet& config);

private:
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkfGeoToken_;

  int level_;
  std::string outputFileName_;
};

DumpMkFitGeometry::DumpMkFitGeometry(const edm::ParameterSet& config)
    : mkfGeoToken_{esConsumes()},
      level_(config.getUntrackedParameter<int>("level", 1)),
      outputFileName_(config.getUntrackedParameter<std::string>("outputFileName", "cmsRecoGeo.root")) {}

void DumpMkFitGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& mkfg = iSetup.getData(mkfGeoToken_);
  const mkfit::TrackerInfo& ti = mkfg.trackerInfo();

  edm::LogInfo("DumpMkFitGeometry") << "geom_ptr=" << &mkfg << "n_layers = " << ti.n_layers();
  if (outputFileName_.empty()) {
    edm::LogInfo("DumpMkFitGeometry") << "no file-name specified, not dumping binary file";
  } else {
    edm::LogInfo("DumpMkFitGeometry") << "binary file = '" << outputFileName_ << "'";
    ti.write_bin_file(outputFileName_);
  }

  ti.print_tracker(level_);
  edm::LogInfo("DumpMkFitGeometry") << "finished, n_modules=" << ti.n_total_modules();
}

DEFINE_FWK_MODULE(DumpMkFitGeometry);
