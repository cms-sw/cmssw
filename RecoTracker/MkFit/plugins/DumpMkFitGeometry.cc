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

  printf("DumpMkFitGeometry geom_ptr=%p, n_layers = %d\n", &mkfg, ti.n_layers());
  if (m_level > 0) {
    int n_modules = 0;
    for (int i = 0; i < ti.n_layers(); ++i) {
      const mkfit::LayerInfo& li = ti.layer(i);
      li.print_layer();
      n_modules += li.n_modules();
      if (m_level > 1) {
        printf("  Detailed module list N=%d\n", li.n_modules());
        for (int j = 0; j < li.n_modules(); ++j) {
          const mkfit::ModuleInfo& mi = li.module_info(j);
          auto* p = mi.m_pos.Array();
          auto* z = mi.m_zdir.Array();
          auto* x = mi.m_xdir.Array();
          // clang-format off
          printf("Layer %d, mid=%u; detid=0x%x pos=%.3f,%.3f,%.3f, "
                 "norm=%.3f,%.3f,%.3f, phi=%.3f,%.3f,%.3f\n",
                 i, j, mi.m_detid, p[0], p[1], p[2],
                 z[0], z[1], z[2], x[0], x[1], x[2]);
          // clang-format on
        }
        printf("\n");
      }
    }
    printf("DumpMkFitGeometry finished, n_modules=%d\n", n_modules);
  }
}

void DumpMkFitGeometry::beginJob(void) {}

void DumpMkFitGeometry::endJob(void) {}

DEFINE_FWK_MODULE(DumpMkFitGeometry);
