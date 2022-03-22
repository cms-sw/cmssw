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
  ~DumpMkFitGeometry(void) override {}

private:
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  void beginJob(void) override;
  void endJob(void) override;

  edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> m_mkfGeoToken;

  int m_level;
  std::string m_tag;
  std::string m_outputFileName;
};

DumpMkFitGeometry::DumpMkFitGeometry(const edm::ParameterSet& config)
    : m_mkfGeoToken{esConsumes()},
      m_level(config.getUntrackedParameter<int>("level", 1)),
      m_tag(config.getUntrackedParameter<std::string>("tagInfo", "unknown")),

      m_outputFileName(config.getUntrackedParameter<std::string>("outputFileName", "cmsRecoGeo.root")) {}

void DumpMkFitGeometry::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& mkfg = iSetup.getData(m_mkfGeoToken);
  const mkfit::TrackerInfo& ti = mkfg.trackerInfo();

  printf("DumpMkFitGeometry geom_ptr=%p, n_layers = %d ==> ", &mkfg, ti.n_layers());
  if (m_outputFileName.empty()) {
    printf("no file-name specified, not dumping binary file\n");
  } else {
    printf("binary file = '%s'\n", m_outputFileName.c_str());
    ti.write_bin_file(m_outputFileName);
  }

  ti.print_tracker(m_level);
  printf("DumpMkFitGeometry finished, n_modules=%d\n", ti.n_total_modules());
}

void DumpMkFitGeometry::beginJob(void) {}

void DumpMkFitGeometry::endJob(void) {}

DEFINE_FWK_MODULE(DumpMkFitGeometry);
