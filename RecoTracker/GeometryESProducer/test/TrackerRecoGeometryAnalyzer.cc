// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

using namespace std;

class TrackerRecoGeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  TrackerRecoGeometryAnalyzer(const edm::ParameterSet&);
  ~TrackerRecoGeometryAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

TrackerRecoGeometryAnalyzer::TrackerRecoGeometryAnalyzer(const edm::ParameterSet& iConfig) {}

TrackerRecoGeometryAnalyzer::~TrackerRecoGeometryAnalyzer() {}

void TrackerRecoGeometryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //
  // get the GeometricSearchDet
  //
  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get(track);

  //---- testing access to barrelLayers ----
  vector<const BarrelDetLayer*> theBarrelLayers = track->barrelLayers();
  edm::LogInfo("TrackerRecoGeometryAnalyzer") << "number of BarrelLayers: " << theBarrelLayers.size();

  for (unsigned int i = 0; i < 3; i++) {
    const BarrelDetLayer* theLayer = theBarrelLayers[i];
    edm::LogInfo("TrackerRecoGeometryAnalyzer")
        << "theLayer[" << i << "]->position().perp(): " << theLayer->components().front()->surface().position().perp();
  }
}

DEFINE_FWK_MODULE(TrackerRecoGeometryAnalyzer);
