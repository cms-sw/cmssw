// system include files
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"
//
// class declaration
//

class GEMDetIdAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit GEMDetIdAnalyzer(const edm::ParameterSet&);
  ~GEMDetIdAnalyzer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  edm::EDGetToken gemRecHit_, me0RecHit_;
  bool newGEM_;
};

GEMDetIdAnalyzer::GEMDetIdAnalyzer(const edm::ParameterSet& iConfig) {
  gemRecHit_ = consumes<GEMRecHitCollection>(iConfig.getParameter<edm::InputTag>("gemInputLabel"));
  me0RecHit_ = consumes<ME0RecHitCollection>(iConfig.getParameter<edm::InputTag>("me0InputLabel"));
  newGEM_ = iConfig.getParameter<bool>("newGEMDetID");
}

void GEMDetIdAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gemInputLabel", edm::InputTag("gemRecHits"));
  desc.add<edm::InputTag>("me0InputLabel", edm::InputTag("me0RecHits"));
  desc.add<bool>("newGEMDetID", true);
  descriptions.add("gemDetIdAnalyzer", desc);
}

void GEMDetIdAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  edm::Handle<GEMRecHitCollection> gemRecHits;
  iEvent.getByToken(gemRecHit_, gemRecHits);
  if (!gemRecHits.isValid()) {
    edm::LogError("GEMAnalysis") << "GEM RecHit is not valid";
  } else {
    edm::LogVerbatim("GEMAnalysis") << "GEMRecHit collection with " << gemRecHits.product()->size() << " hits";
    unsigned int k(0);
    for (const auto& hit : *(gemRecHits.product())) {
      GEMDetId id = hit.gemId();
      edm::LogVerbatim("GEMAnalysis") << "Hit[" << k << "] " << std::hex << id.rawId() << std::dec << " " << id;
      ++k;
    }
  }

  edm::Handle<ME0RecHitCollection> me0RecHits;
  iEvent.getByToken(me0RecHit_, me0RecHits);
  if (!me0RecHits.isValid()) {
    edm::LogError("GEMAnalysis") << "ME0 RecHit is not valid";
  } else {
    edm::LogVerbatim("GEMAnalysis") << "ME0RecHit collection with " << me0RecHits.product()->size() << " hits";
    unsigned int k(0);
    for (const auto& hit : *(me0RecHits.product())) {
      ME0DetId id = hit.me0Id();
      edm::LogVerbatim("GEMAnalysis") << "Hit[" << k << "] " << std::hex << id.rawId() << std::dec << " " << id;
      if (newGEM_) {
        GEMDetId idx(id.rawId());
        edm::LogVerbatim("GEMAnalysis") << "for GEM " << std::hex << idx.rawId() << std::dec << " " << idx;
      }

      ++k;
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(GEMDetIdAnalyzer);
