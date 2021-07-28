#include "DQM/GEM/interface/GEMRecHitSource.h"

using namespace std;
using namespace edm;

GEMRecHitSource::GEMRecHitSource(const edm::ParameterSet& cfg) : GEMDQMBase(cfg) {
  tagRecHit_ = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitsInputLabel"));

  nIdxFirstDigi_ = cfg.getParameter<int>("idxFirstDigi");
  nClusterSizeBinNum_ = cfg.getParameter<int>("ClusterSizeBinNum");
  bModeRelVal_ = cfg.getParameter<bool>("modeRelVal");
}

void GEMRecHitSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHitsInputLabel", edm::InputTag("gemRecHits", ""));

  desc.add<int>("idxFirstDigi", 0);
  desc.add<int>("ClusterSizeBinNum", 9);
  desc.add<bool>("modeRelVal", false);

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
  ibooker.setCurrentFolder("GEM/RecHits");

  nCLSMax_ = 10;
  fRadiusMin_ = 120.0;
  fRadiusMax_ = 250.0;
  float radS = -5.0 / 180 * 3.141592;
  float radL = 355.0 / 180 * 3.141592;

  mapTotalRecHit_layer_ = MEMap3Inf(this, "det", "RecHit Occupancy", 36, 0.5, 36.5, 8, 0.5, 8.5, "Chamber", "iEta");
  mapRecHitWheel_layer_ = MEMap3Inf(
      this, "rphi_occ", "RecHit R-Phi Occupancy", 108, radS, radL, 8, fRadiusMin_, fRadiusMax_, "#phi (rad)", "R [cm]");
  mapRecHitOcc_ieta_ = MEMap3Inf(this, "occ_ieta", "RecHit iEta Occupancy", 8, 0.5, 8.5, "iEta", "Number of RecHits");
  mapRecHitOcc_phi_ =
      MEMap3Inf(this, "occ_phi", "RecHit Phi Occupancy", 108, -5, 355, "#phi (degree)", "Number of RecHits");
  mapTotalRecHitPerEvtLayer_ = MEMap3Inf(this,
                                         "rechits_per_layer",
                                         "Total number of RecHits per event for each layers",
                                         50,
                                         -0.5,
                                         99.5,
                                         "Number of RecHits",
                                         "Events");
  mapTotalRecHitPerEvtIEta_ = MEMap3Inf(this,
                                        "rechits_per_ieta",
                                        "Total number of RecHits per event for each eta partitions",
                                        50,
                                        -0.5,
                                        99.5,
                                        "Number of RecHits",
                                        "Events");
  mapCLSRecHit_ieta_ = MEMap3Inf(
      this, "cls", "Cluster size of RecHits", nCLSMax_, 0.5, nCLSMax_ + 0.5, "Cluster size", "Number of RecHits");
  mapCLSAverage_ = MEMap3Inf(this,  // TProfile2D
                             "rechit_average",
                             "Average of Cluster Sizes",
                             36,
                             0.5,
                             36.5,
                             8,
                             0.5,
                             8.5,
                             0,
                             400,  // For satefy, larger than 384
                             "Chamber",
                             "iEta");
  mapCLSOver5_ = MEMap3Inf(
      this, "largeCls_occ", "Occupancy of Large Clusters (>5)", 36, 0.5, 36.5, 8, 0.5, 8.5, "Chamber", "iEta");

  mapCLSPerCh_ = MEMap4Inf(
      this, "cls", "Cluster size of RecHits", nCLSMax_, 0.5, nCLSMax_ + 0.5, 1, 0.5, 1.5, "Cluster size", "iEta");

  if (bModeRelVal_) {
    mapTotalRecHit_layer_.TurnOff();
    mapRecHitWheel_layer_.TurnOff();
    mapCLSAverage_.TurnOff();
    mapCLSOver5_.TurnOff();
    mapCLSPerCh_.TurnOff();
  }

  GenerateMEPerChamber(ibooker);
}

int GEMRecHitSource::ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) {
  mapTotalRecHitPerEvtIEta_.bookND(bh, key);

  return 0;
}

int GEMRecHitSource::ProcessWithMEMap2AbsReWithEta(BookingHelper& bh, ME3IdsKey key) {
  mapCLSRecHit_ieta_.bookND(bh, key);

  return 0;
}

int GEMRecHitSource::ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) {
  MEStationInfo& stationInfo = mapStationInfo_[key];

  Int_t nNumVFATPerEta = stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_;

  mapTotalRecHit_layer_.SetBinConfX(stationInfo.nNumChambers_);
  mapTotalRecHit_layer_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapTotalRecHit_layer_.bookND(bh, key);
  mapTotalRecHit_layer_.SetLabelForChambers(key, 1);
  mapTotalRecHit_layer_.SetLabelForIEta(key, 2);

  mapRecHitWheel_layer_.SetBinLowEdgeX(-0.088344);  // FIXME: It could be different for other stations...
  mapRecHitWheel_layer_.SetBinHighEdgeX(-0.088344 + 2 * 3.141592);
  mapRecHitWheel_layer_.SetNbinsX(nNumVFATPerEta * stationInfo.nNumChambers_);
  mapRecHitWheel_layer_.SetNbinsY(stationInfo.nNumEtaPartitions_);
  mapRecHitWheel_layer_.bookND(bh, key);

  mapRecHitOcc_ieta_.SetBinConfX(stationInfo.nNumEtaPartitions_);
  mapRecHitOcc_ieta_.bookND(bh, key);
  mapRecHitOcc_ieta_.SetLabelForIEta(key, 1);

  mapRecHitOcc_phi_.bookND(bh, key);
  mapTotalRecHitPerEvtLayer_.bookND(bh, key);

  mapCLSAverage_.bookND(bh, key);
  mapCLSOver5_.bookND(bh, key);
  mapCLSAverage_.SetLabelForChambers(key, 1);
  mapCLSAverage_.SetLabelForIEta(key, 2);
  mapCLSOver5_.SetLabelForChambers(key, 1);
  mapCLSOver5_.SetLabelForIEta(key, 2);

  return 0;
}

int GEMRecHitSource::ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  mapCLSPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapCLSPerCh_.bookND(bh, key);
  mapCLSPerCh_.SetLabelForIEta(key, 2);

  return 0;
}

void GEMRecHitSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<GEMRecHitCollection> gemRecHits;
  event.getByToken(this->tagRecHit_, gemRecHits);
  if (!gemRecHits.isValid()) {
    edm::LogError(log_category_) << "GEM RecHit is not valid.\n";
    return;
  }

  std::map<ME3IdsKey, Int_t> total_rechit_layer;
  std::map<ME3IdsKey, Int_t> total_rechit_iEta;
  std::map<ME4IdsKey, std::map<Int_t, Bool_t>> mapCLSOver5;

  for (const auto& ch : gemChambers_) {
    GEMDetId gid = ch.id();
    auto chamber = gid.chamber();
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
    MEStationInfo& stationInfo = mapStationInfo_[key3];
    for (auto ieta : ch.etaPartitions()) {
      GEMDetId eId = ieta->id();
      ME3IdsKey key3IEta{gid.region(), gid.station(), eId.ieta()};
      ME3IdsKey key3AbsReIEta{std::abs(gid.region()), gid.station(), eId.ieta()};
      ME4IdsKey key4IEta{gid.region(), gid.station(), gid.layer(), eId.ieta()};

      if (total_rechit_layer.find(key3) == total_rechit_layer.end())
        total_rechit_layer[key3] = 0;

      const auto& recHitsRange = gemRecHits->get(eId);
      auto gemRecHit = recHitsRange.first;
      for (auto hit = gemRecHit; hit != recHitsRange.second; ++hit) {
        GlobalPoint recHitGP = GEMGeometry_->idToDet(hit->gemId())->surface().toGlobal(hit->localPosition());
        Float_t fPhi = recHitGP.phi();
        if (fPhi < -5.0 / 180.0 * 3.141592)
          fPhi += 2 * 3.141592;

        // Filling of RecHit occupancy
        mapTotalRecHit_layer_.Fill(key3, chamber, eId.ieta());

        // Filling of R-Phi occupancy
        Float_t fR = fRadiusMin_ + (fRadiusMax_ - fRadiusMin_) * (eId.ieta() - 0.5) / stationInfo.nNumEtaPartitions_;
        mapRecHitWheel_layer_.Fill(key3, fPhi, fR);

        // Filling of RecHit (iEta)
        mapRecHitOcc_ieta_.Fill(key3, eId.ieta());

        // Filling of RecHit (phi)
        Float_t fPhiDeg = fPhi * 180.0 / 3.141592;
        mapRecHitOcc_phi_.Fill(key3, fPhiDeg);

        // For total RecHits
        total_rechit_layer[key3]++;
        total_rechit_iEta[key3IEta]++;

        // Filling of cluster size (CLS)
        Int_t nCLS = hit->clusterSize();
        Int_t nCLSCutOff = std::min(nCLS, nCLSMax_);  // For overflow
        mapCLSRecHit_ieta_.Fill(key3AbsReIEta, nCLSCutOff);
        mapCLSPerCh_.Fill(key4Ch, nCLSCutOff, eId.ieta());
        mapCLSAverage_.Fill(key3, (Double_t)chamber, (Double_t)eId.ieta(), nCLS);
        if (nCLS > 5)
          mapCLSOver5[key4IEta][chamber] = true;
      }
    }
  }
  for (auto [key, num_total_rechit] : total_rechit_layer) {
    mapTotalRecHitPerEvtLayer_.Fill(key, num_total_rechit);
  }
  for (auto [key, num_total_rechit] : total_rechit_iEta) {
    mapTotalRecHitPerEvtIEta_.Fill(key, num_total_rechit);
  }
  for (auto [key, mapSub] : mapCLSOver5) {
    for (auto [chamber, b] : mapSub) {
      mapCLSOver5_.Fill(key4Tokey3(key), chamber, keyToIEta(key));
    }
  }
}

DEFINE_FWK_MODULE(GEMRecHitSource);
