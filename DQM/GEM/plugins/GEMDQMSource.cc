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
#include "DQMServices/Core/interface/MonitorElement.h"

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

private:
  edm::EDGetToken tagRecHit_;

  float fGlobXMin_, fGlobXMax_;
  float fGlobYMin_, fGlobYMax_;

  int nIdxFirstStrip_;

  const GEMGeometry* initGeometry(edm::EventSetup const& iSetup);
  int findVFAT(float min_, float max_, float x_, int roll_);

  const GEMGeometry* GEMGeometry_;

  std::vector<GEMChamber> gemChambers_;

  std::unordered_map<UInt_t, MonitorElement*> recHitME_;
  std::unordered_map<UInt_t, MonitorElement*> VFAT_vs_ClusterSize_;
  std::unordered_map<UInt_t, MonitorElement*> StripsFired_vs_eta_;
  std::unordered_map<UInt_t, MonitorElement*> rh_vs_eta_;
  std::unordered_map<UInt_t, MonitorElement*> recGlobalPos;
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

  nIdxFirstStrip_ = cfg.getParameter<int>("idxFirstStrip");

  fGlobXMin_ = cfg.getParameter<double>("global_x_bound_min");
  fGlobXMax_ = cfg.getParameter<double>("global_x_bound_max");
  fGlobYMin_ = cfg.getParameter<double>("global_y_bound_min");
  fGlobYMax_ = cfg.getParameter<double>("global_y_bound_max");
}

void GEMDQMSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHitsInputLabel", edm::InputTag("gemRecHits", ""));

  desc.add<int>("idxFirstStrip", 0);

  desc.add<double>("global_x_bound_min", -350);
  desc.add<double>("global_x_bound_max", 350);
  desc.add<double>("global_y_bound_min", -260);
  desc.add<double>("global_y_bound_max", 260);

  descriptions.add("GEMDQMSource", desc);
}

void GEMDQMSource::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  std::vector<GEMDetId> listLayerOcc;

  GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;

  const std::vector<const GEMSuperChamber*>& superChambers_ = GEMGeometry_->superChambers();
  for (auto sch : superChambers_) {
    int n_lay = sch->nChambers();
    for (int l = 0; l < n_lay; l++) {
      Bool_t bExist = false;
      for (auto ch : gemChambers_)
        if (ch.id() == sch->chamber(l + 1)->id())
          bExist = true;
      if (bExist)
        continue;

      gemChambers_.push_back(*sch->chamber(l + 1));
    }
  }

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/recHit");

  for (auto ch : gemChambers_) {
    GEMDetId gid = ch.id();

    std::string strIdxName = "Gemini_" + to_string(gid.chamber()) + "_GE" + (gid.region() > 0 ? "p" : "m") +
                             to_string(gid.station()) + "_" + to_string(gid.layer());
    std::string strIdxTitle = "GEMINIm" + to_string(gid.chamber()) + " in GE" + (gid.region() > 0 ? "+" : "-") +
                              to_string(gid.station()) + "/" + to_string(gid.layer());

    std::string strStId = (gid.region() > 0 ? "p" : "m") + std::to_string(gid.station());
    std::string strStT = (gid.region() > 0 ? "+" : "-") + std::to_string(gid.station());

    std::string strLa = std::to_string(gid.layer());
    std::string strCh = std::to_string(gid.chamber());

    string hName = "recHit_" + strIdxName;
    string hTitle = "recHit " + strIdxTitle;
    hTitle += ";VFAT;";
    recHitME_[gid] = ibooker.book1D(hName, hTitle, 24, 0, 24);

    string hName_2 = "VFAT_vs_ClusterSize_" + strIdxName;
    string hTitle_2 = "VFAT vs ClusterSize " + strIdxTitle;
    hTitle_2 += ";Cluster size;VFAT";
    VFAT_vs_ClusterSize_[gid] = ibooker.book2D(hName_2, hTitle_2, 9, 1, 10, 24, 0, 24);

    string hName_fired = "StripFired_" + strIdxName;
    string hTitle_fired = "StripsFired " + strIdxTitle;
    hTitle_fired += ";Strip;iEta";
    StripsFired_vs_eta_[gid] = ibooker.book2D(hName_fired, hTitle_fired, 384, 1, 385, 8, 1, 9);

    string hName_rh = "recHit_x_" + strIdxName;
    string hTitle_rh = "recHit local x " + strIdxTitle;
    hTitle_rh += ";Local x (cm);iEta";
    rh_vs_eta_[gid] = ibooker.book2D(hName_rh, hTitle_rh, 50, -25, 25, 8, 1, 9);

    GEMDetId lid(gid.region(), gid.ring(), gid.station(), gid.layer(), 0, 0);
    Int_t nIdxOcc = 0;

    for (; nIdxOcc < (Int_t)listLayerOcc.size(); nIdxOcc++)
      if (listLayerOcc[nIdxOcc] == lid)
        break;
    if (nIdxOcc >= (Int_t)listLayerOcc.size())
      listLayerOcc.push_back(lid);
  }

  for (auto gid : listLayerOcc) {
    std::string strIdxName =
        std::string("GE") + (gid.region() > 0 ? "p" : "m") + to_string(gid.station()) + "_" + to_string(gid.layer());
    std::string strIdxTitle =
        std::string("GE") + (gid.region() > 0 ? "+" : "-") + to_string(gid.station()) + "/" + to_string(gid.layer());

    string hName_rh = "recHit_globalPos_Gemini_" + strIdxName;
    string hTitle_rh = "recHit global position Gemini chamber : " + strIdxTitle;
    hTitle_rh += ";Global X (cm);Global Y (cm)";
    recGlobalPos[gid] = ibooker.book2D(hName_rh, hTitle_rh, 100, fGlobXMin_, fGlobXMax_, 100, fGlobYMin_, fGlobYMax_);
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
        //int nVfat = findVFAT(0.0, 384.0, hit->firstClusterStrip()+0.5*hit->clusterSize(), rId.roll());
        Int_t nIdxStrip = hit->firstClusterStrip() + 0.5 * hit->clusterSize() - nIdxFirstStrip_;
        Int_t nVfat = 8 * ((Int_t)(nIdxStrip / (roll->nstrips() / 3)) + 1) - rId.roll();  // Strip:Start at 0
        recHitME_[cId]->Fill(nVfat);
        rh_vs_eta_[cId]->Fill(hit->localPosition().x(), rId.roll());
        VFAT_vs_ClusterSize_[cId]->Fill(hit->clusterSize(), nVfat);
        for (int i = hit->firstClusterStrip(); i < (hit->firstClusterStrip() + hit->clusterSize()); i++) {
          StripsFired_vs_eta_[cId]->Fill(i, rId.roll());
        }

        GlobalPoint recHitGP = GEMGeometry_->idToDet(hit->gemId())->surface().toGlobal(hit->localPosition());
        GEMDetId idLayer(rId.region(), rId.ring(), rId.station(), rId.layer(), 0, 0);
        recGlobalPos[idLayer]->Fill(recHitGP.x(), recHitGP.y());
      }
    }
  }
}

DEFINE_FWK_MODULE(GEMDQMSource);
