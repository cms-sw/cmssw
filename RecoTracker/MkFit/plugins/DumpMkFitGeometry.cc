#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

#include "TrackerInfo.h"

class DumpMkFitGeometry : public edm::EDAnalyzer {
public:
  explicit DumpMkFitGeometry(const edm::ParameterSet& config);
  ~DumpMkFitGeometry(void) override {}

private:
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  void beginJob(void) override;
  void endJob(void) override;

  int m_level;
  std::string m_tag;
  std::string m_outputFileName;
};

DumpMkFitGeometry::DumpMkFitGeometry(const edm::ParameterSet& config)
  : m_level(config.getUntrackedParameter<int>("level", 1)),
    m_tag(config.getUntrackedParameter<std::string>("tagInfo", "unknown")),

    m_outputFileName(config.getUntrackedParameter<std::string>("outputFileName", "cmsRecoGeo.root"))
{}

void DumpMkFitGeometry::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  using namespace edm;

  ESTransientHandle<MkFitGeometry> geoh;
  eventSetup.get<TrackerRecoGeometryRecord>().get(geoh);

  const MkFitGeometry *mkfg = geoh.product();

  printf("MkFitGeompetry ptr = %p\n", mkfg);

  mkfit::TrackerInfo const &ti = mkfg->trackerInfo();

  int nl = ti.m_layers.size();
  for (int i = 0; i < nl; ++i)
  {
    const_cast<mkfit::LayerInfo&>(ti.m_layers[i]).print_layer();
  }
}

void DumpMkFitGeometry::beginJob(void) {}

void DumpMkFitGeometry::endJob(void) {}


DEFINE_FWK_MODULE(DumpMkFitGeometry);
