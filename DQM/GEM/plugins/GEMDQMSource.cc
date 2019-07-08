#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMDQMSource : public DQMEDAnalyzer {
public:
  GEMDQMSource(const edm::ParameterSet& cfg);
  ~GEMDQMSource() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override{};
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override{};

private:
  edm::EDGetToken tagRecHit_;

  const GEMGeometry* initGeometry(edm::EventSetup const& iSetup);
  int findVFAT(float min_, float max_, float x_, int roll_);

  const GEMGeometry* GEMGeometry_;

  std::vector<GEMChamber> gemChambers_;

  std::unordered_map<UInt_t, MonitorElement*> recHitME_;
  std::unordered_map<UInt_t, MonitorElement*> VFAT_vs_ClusterSize_;
  std::unordered_map<UInt_t, MonitorElement*> StripsFired_vs_eta_;
  std::unordered_map<UInt_t, MonitorElement*> rh_vs_eta_;
};

using namespace std;
using namespace edm;

int GEMDQMSource::findVFAT(float min_, float max_, float x_, int roll_) {
  float step = abs(max_ - min_) / 3.0;
  if (x_ < (min(min_, max_) + step)) {
    return 8 - roll_;
  } else if (x_ < (min(min_, max_) + 2.0 * step)) {
    return 16 - roll_;
  } else {
    return 24 - roll_;
  }
}

const GEMGeometry* GEMDQMSource::initGeometry(edm::EventSetup const& iSetup) {
  const GEMGeometry* GEMGeometry_ = nullptr;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  } catch (edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("MuonGEMBaseValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return nullptr;
  }
  return GEMGeometry_;
}

GEMDQMSource::GEMDQMSource(const edm::ParameterSet& cfg) {
  tagRecHit_ = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitsInputLabel"));
}

void GEMDQMSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHitsInputLabel", edm::InputTag("gemRecHits", ""));
  descriptions.add("GEMDQMSource", desc);
}

void GEMDQMSource::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;

  const std::vector<const GEMSuperChamber*>& superChambers_ = GEMGeometry_->superChambers();
  for (auto sch : superChambers_) {
    int n_lay = sch->nChambers();
    for (int l = 0; l < n_lay; l++) {
      gemChambers_.push_back(*sch->chamber(l + 1));
    }
  }
  ibooker.cd();
  ibooker.setCurrentFolder("GEM/recHit");
  for (auto ch : gemChambers_) {
    GEMDetId gid = ch.id();
    string hName = "recHit_Gemini_" + to_string(gid.chamber()) + "_la_" + to_string(gid.layer());
    string hTitle = "recHit Gemini chamber : " + to_string(gid.chamber()) + ", layer : " + to_string(gid.layer());
    recHitME_[ch.id()] = ibooker.book1D(hName, hTitle, 24, 0, 24);

    string hName_2 = "VFAT_vs_ClusterSize_Gemini_" + to_string(gid.chamber()) + "_la_" + to_string(gid.layer());
    string hTitle_2 =
        "VFAT vs ClusterSize Gemini chamber : " + to_string(gid.chamber()) + ", layer : " + to_string(gid.layer());
    VFAT_vs_ClusterSize_[ch.id()] = ibooker.book2D(hName_2, hTitle_2, 9, 1, 10, 24, 0, 24);

    string hName_fired = "StripFired_Gemini_" + to_string(gid.chamber()) + "_la_" + to_string(gid.layer());
    string hTitle_fired =
        "StripsFired Gemini chamber : " + to_string(gid.chamber()) + ", layer : " + to_string(gid.layer());
    StripsFired_vs_eta_[ch.id()] = ibooker.book2D(hName_fired, hTitle_fired, 384, 1, 385, 8, 1, 9);

    string hName_rh = "recHit_x_Gemini_" + to_string(gid.chamber()) + "_la_" + to_string(gid.layer());
    string hTitle_rh =
        "recHit local x Gemini chamber : " + to_string(gid.chamber()) + ", layer : " + to_string(gid.layer());
    rh_vs_eta_[ch.id()] = ibooker.book2D(hName_rh, hTitle_rh, 50, -25, 25, 8, 1, 9);
  }
}

void GEMDQMSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  const GEMGeometry* GEMGeometry_ = initGeometry(eventSetup);
  if (GEMGeometry_ == nullptr)
    return;

  edm::Handle<GEMRecHitCollection> gemRecHits;
  event.getByToken(this->tagRecHit_, gemRecHits);
  if (!gemRecHits.isValid()) {
    edm::LogError("GEMDQMSource") << "GEM recHit is not valid.\n";
    return;
  }
  for (auto ch : gemChambers_) {
    GEMDetId cId = ch.id();
    for (auto roll : ch.etaPartitions()) {
      GEMDetId rId = roll->id();
      const auto& recHitsRange = gemRecHits->get(rId);
      auto gemRecHit = recHitsRange.first;
      for (auto hit = gemRecHit; hit != recHitsRange.second; ++hit) {
        int nVfat = findVFAT(1.0, 385.0, hit->firstClusterStrip() + 0.5 * hit->clusterSize(), rId.roll());
        recHitME_[cId]->Fill(nVfat);
        rh_vs_eta_[cId]->Fill(hit->localPosition().x(), rId.roll());
        VFAT_vs_ClusterSize_[cId]->Fill(hit->clusterSize(), nVfat);
        for (int i = hit->firstClusterStrip(); i < (hit->firstClusterStrip() + hit->clusterSize()); i++) {
          StripsFired_vs_eta_[cId]->Fill(i, rId.roll());
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(GEMDQMSource);
