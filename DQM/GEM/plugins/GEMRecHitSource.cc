#include "DQM/GEM/interface/GEMDQMBase.h"

#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMRecHitSource : public GEMDQMBase {
public:
  explicit GEMRecHitSource(const edm::ParameterSet& cfg);
  ~GEMRecHitSource() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override{};
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  int ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap4(BookingHelper& bh, ME4IdsKey key) override;
  int ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) override;

  edm::EDGetToken tagRecHit_;

  float fGlobXMin_, fGlobXMax_;
  float fGlobYMin_, fGlobYMax_;

  int nIdxFirstStrip_;
  int nClusterSizeBinNum_;

  MEMap3Inf mapTotalRecHit_layer_;
  MEMap3Inf mapRecHitOcc_ieta_;
  MEMap3Inf mapRecHitOcc_phi_;
  MEMap3Inf mapTotalRecHitPerEvt_;
  MEMap4Inf mapCLSRecHit_ieta_;

  MEMap4Inf mapCLSPerCh_;

  Int_t nCLSMax_;

  std::unordered_map<UInt_t, MonitorElement*> recHitME_;
  std::unordered_map<UInt_t, MonitorElement*> VFAT_vs_ClusterSize_;
  std::unordered_map<UInt_t, MonitorElement*> StripsFired_vs_eta_;
  std::unordered_map<UInt_t, MonitorElement*> rh_vs_eta_;
  std::unordered_map<UInt_t, MonitorElement*> recGlobalPos;
};

using namespace std;
using namespace edm;

GEMRecHitSource::GEMRecHitSource(const edm::ParameterSet& cfg) : GEMDQMBase(cfg) {
  tagRecHit_ = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitsInputLabel"));

  nIdxFirstStrip_ = cfg.getParameter<int>("idxFirstStrip");
  nClusterSizeBinNum_ = cfg.getParameter<int>("ClusterSizeBinNum");

  fGlobXMin_ = cfg.getParameter<double>("global_x_bound_min");
  fGlobXMax_ = cfg.getParameter<double>("global_x_bound_max");
  fGlobYMin_ = cfg.getParameter<double>("global_y_bound_min");
  fGlobYMax_ = cfg.getParameter<double>("global_y_bound_max");
}

void GEMRecHitSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHitsInputLabel", edm::InputTag("gemRecHits", ""));

  desc.add<int>("idxFirstStrip", 0);
  desc.add<int>("ClusterSizeBinNum", 9);

  desc.add<double>("global_x_bound_min", -350);
  desc.add<double>("global_x_bound_max", 350);
  desc.add<double>("global_y_bound_min", -260);
  desc.add<double>("global_y_bound_max", 260);

  desc.addUntracked<std::string>("logCategory", "GEMRecHitSource");

  descriptions.add("GEMRecHitSource", desc);
}

void GEMRecHitSource::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  std::vector<GEMDetId> listLayerOcc;

  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/recHit");

  nCLSMax_ = 10;

  mapTotalRecHit_layer_ =
      MEMap3Inf(this, "rechit_det", "Rec. Hit Occupancy", 36, 0.5, 36.5, 8, 0.5, 8.5, "Chamber", "iEta");
  mapRecHitOcc_ieta_ = MEMap3Inf(
      this, "rechit_ieta_occ", "RecHit Digi Occupancy per eta partition", 8, 0.5, 8.5, "iEta", "Number of recHits");
  mapRecHitOcc_phi_ = MEMap3Inf(
      this, "rechit_phi_occ", "RecHit Digi Phi (degree) Occupancy ", 108, -5, 355, "#phi (degree)", "Number of recHits");
  mapTotalRecHitPerEvt_ = MEMap3Inf(this,
                                    "total_rechit_per_event",
                                    "Total number of rec. hits per event for each layers ",
                                    50,
                                    -0.5,
                                    99.5,
                                    "Number of recHits",
                                    "Events");
  mapCLSRecHit_ieta_ = MEMap4Inf(
      this, "cls", "Cluster size of rec. hits", nCLSMax_, 0.5, nCLSMax_ + 0.5, "Cluster size", "Number of recHits");

  mapCLSPerCh_ = MEMap4Inf(
      this, "cls", "Cluster size of rec. hits", nCLSMax_, 0.5, nCLSMax_ + 0.5, 1, 0.5, 1.5, "Cluster size", "iEta");

  GenerateMEPerChamber(ibooker);
}

int GEMRecHitSource::ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) {
  MEStationInfo& stationInfo = mapStationInfo_[key];

  mapTotalRecHit_layer_.SetBinConfX(stationInfo.nNumChambers_);
  mapTotalRecHit_layer_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapTotalRecHit_layer_.bookND(bh, key);
  mapTotalRecHit_layer_.SetLabelForChambers(key, 1);
  mapTotalRecHit_layer_.SetLabelForChambers(key, 2);  // No worries, it's same as the eta partition labelling

  mapRecHitOcc_ieta_.SetBinConfX(stationInfo.nNumEtaPartitions_);
  mapRecHitOcc_ieta_.bookND(bh, key);

  mapRecHitOcc_phi_.bookND(bh, key);
  mapTotalRecHitPerEvt_.bookND(bh, key);

  return 0;
}

int GEMRecHitSource::ProcessWithMEMap4(BookingHelper& bh, ME4IdsKey key) {
  mapCLSRecHit_ieta_.bookND(bh, key);

  return 0;
}

int GEMRecHitSource::ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  mapCLSPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapCLSPerCh_.bookND(bh, key);
  mapCLSPerCh_.SetLabelForChambers(key, 2);  // For eta partitions

  return 0;
}

void GEMRecHitSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<GEMRecHitCollection> gemRecHits;
  event.getByToken(this->tagRecHit_, gemRecHits);
  if (!gemRecHits.isValid()) {
    edm::LogError(log_category_) << "GEM recHit is not valid.\n";
    return;
  }
  std::map<ME3IdsKey, Int_t> total_rechit_layer;
  for (const auto& ch : gemChambers_) {
    GEMDetId gid = ch.id();
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
    for (auto ieta : ch.etaPartitions()) {
      GEMDetId rId = ieta->id();
      ME4IdsKey key4iEta{gid.region(), gid.station(), gid.layer(), rId.roll()};
      if (total_rechit_layer.find(key3) == total_rechit_layer.end())
        total_rechit_layer[key3] = 0;
      const auto& recHitsRange = gemRecHits->get(rId);
      auto gemRecHit = recHitsRange.first;
      for (auto hit = gemRecHit; hit != recHitsRange.second; ++hit) {
        // Filling of digi occupancy
        mapTotalRecHit_layer_.Fill(key3, gid.chamber(), rId.roll());

        // Filling of strip (iEta)
        mapRecHitOcc_ieta_.Fill(key3, rId.roll());

        // Filling of strip (phi)
        GlobalPoint recHitGP = GEMGeometry_->idToDet(hit->gemId())->surface().toGlobal(hit->localPosition());
        Float_t fPhiDeg = ((Float_t)recHitGP.phi()) * 180.0 / 3.141592;
        if (fPhiDeg < -5.0)
          fPhiDeg += 360.0;
        mapRecHitOcc_phi_.Fill(key3, fPhiDeg);

        // For total recHits
        total_rechit_layer[key3]++;

        // Filling of cluster size (CLS)
        Int_t nCLS = std::min(hit->clusterSize(), nCLSMax_);  // For overflow
        mapCLSRecHit_ieta_.Fill(key4iEta, nCLS);
        mapCLSPerCh_.Fill(key4Ch, rId.roll(), nCLS);
      }
    }
  }
  for (auto [key, num_total_rechit] : total_rechit_layer)
    mapTotalRecHitPerEvt_.Fill(key, num_total_rechit);
}

DEFINE_FWK_MODULE(GEMRecHitSource);
