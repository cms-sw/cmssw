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

  mapTotalRecHit_layer_ =
      MEMap3Inf(this, "det", "RecHit Occupancy", 36, 0.5, 36.5, 8, 0.5, 8.5, "Chamber", "iEta");
  mapRecHitWheel_layer_ =
      MEMap3Inf(this, "wheel", "RecHit Wheel Occupancy", 108, radS, radL, 8, fRadiusMin_, fRadiusMax_, "", "");
  mapRecHitOcc_ieta_ = MEMap3Inf(
      this, "occ_ieta", "RecHit Digi Occupancy per eta partition", 8, 0.5, 8.5, "iEta", "Number of RecHits");
  mapRecHitOcc_phi_ = MEMap3Inf(
      this, "occ_phi", "RecHit Digi Phi (degree) Occupancy", 108, -5, 355, "#phi (degree)", "Number of RecHits");
  mapTotalRecHitPerEvtLayer_ = MEMap3Inf(this,
                                         "total_rechit_per_event",
                                         "Total number of RecHits per event for each layers",
                                         50,
                                         -0.5,
                                         99.5,
                                         "Number of RecHits",
                                         "Events");
  mapTotalRecHitPerEvtIEta_ = MEMap3Inf(this,
                                        "total_rechit_per_event",
                                        "Total number of RecHits per event for each eta partitions",
                                        50,
                                        -0.5,
                                        99.5,
                                        "Number of RecHits",
                                        "Events");
  mapCLSRecHit_ieta_ = MEMap3Inf(
      this, "cls", "Cluster size of RecHits", nCLSMax_, 0.5, nCLSMax_ + 0.5, "Cluster size", "Number of RecHits");
  mapCLSNumberAve_ = MEMap3Inf(this, 
                               "rechitNumberAve", 
                               "Number of RecHits as a function of chambers vs iEta", 
                               36, 
                               0.5, 
                               36.5, 
                               8, 
                               0.5, 
                               8.5, 
                               "Chamber", 
                               "iEta");
  mapCLSNumberOv5_ = MEMap3Inf(this, 
                               "rechitNumberOv5", 
                               "Number of RecHits with cluster size>5 as a function of chambers vs iEta", 
                               36, 
                               0.5, 
                               36.5, 
                               8, 
                               0.5, 
                               8.5, 
                               "Chamber", 
                               "iEta");
  mapCLSAverage_   = MEMap3Inf(this, 
                               "rechit_average_pre", 
                               "Average of Cluster Sizes", 
                               36, 
                               0.5, 
                               36.5, 
                               8, 
                               0.5, 
                               8.5, 
                               "Chamber", 
                               "iEta");
  mapCLSOver5_     = MEMap3Inf(this, 
                               "rechit_over5_pre", 
                               "Average of Cluster Sizes (> 5)", 
                               36, 
                               0.5, 
                               36.5, 
                               8, 
                               0.5, 
                               8.5, 
                               "Chamber", 
                               "iEta");

  mapCLSPerCh_ = MEMap4Inf(
      this, "cls", "Cluster size of RecHits", nCLSMax_, 0.5, nCLSMax_ + 0.5, 1, 0.5, 1.5, "Cluster size", "iEta");

  if ( bModeRelVal_ ) {
    mapTotalRecHit_layer_.TurnOff();
    mapRecHitWheel_layer_.TurnOff();
    mapCLSNumberAve_.TurnOff();
    mapCLSNumberOv5_.TurnOff();
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

  mapRecHitOcc_phi_.bookND(bh, key);
  mapTotalRecHitPerEvtLayer_.bookND(bh, key);
  mapTotalRecHitPerEvtIEta_.bookND(bh, key);

  mapCLSNumberAve_.bookND(bh, key);
  mapCLSNumberOv5_.bookND(bh, key);
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
  ME5map mapCLSNumberAve;
  ME5map mapCLSNumberOv5;
  ME5map mapCLSAverage;
  ME5map mapCLSOver5;

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

      if (total_rechit_layer.find(key3) == total_rechit_layer.end()) total_rechit_layer[key3] = 0;
      InitKey5MultiMap(mapCLSNumberAve, key4IEta, chamber);
      InitKey5MultiMap(mapCLSNumberOv5, key4IEta, chamber);
      InitKey5MultiMap(mapCLSAverage,   key4IEta, chamber);
      InitKey5MultiMap(mapCLSOver5,     key4IEta, chamber);

      const auto& recHitsRange = gemRecHits->get(eId);
      auto gemRecHit = recHitsRange.first;
      Int_t nNumRecHitAve = 0, nNumRecHitOv5 = 0;
      for (auto hit = gemRecHit; hit != recHitsRange.second; ++hit) {
        GlobalPoint recHitGP = GEMGeometry_->idToDet(hit->gemId())->surface().toGlobal(hit->localPosition());
        Float_t fPhi = recHitGP.phi();
        if (fPhi < -5.0 / 180.0 * 3.141592)
          fPhi += 2 * 3.141592;

        // Filling of RecHit occupancy
        mapTotalRecHit_layer_.Fill(key3, chamber, eId.ieta());

        // Filling of wheel occupancy
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
        mapCLSPerCh_.Fill(key4Ch, eId.ieta(), nCLSCutOff);
        mapCLSAverage[key4IEta][chamber] += nCLS;
        nNumRecHitAve++;
        if ( nCLS > 5 ) {
          mapCLSOver5[key4IEta][chamber] += nCLS;
          nNumRecHitOv5++;
        }
      }
      mapCLSNumberAve[key4IEta][chamber] = nNumRecHitAve;
      mapCLSNumberOv5[key4IEta][chamber] = nNumRecHitOv5;
    }
  }
  for (auto [key, num_total_rechit] : total_rechit_layer) {
    mapTotalRecHitPerEvtLayer_.Fill(key, num_total_rechit);
  }
  for (auto [key, num_total_rechit] : total_rechit_iEta) {
    mapTotalRecHitPerEvtIEta_.Fill(key, num_total_rechit);
  }
  for (auto [key, mapSub] : mapCLSNumberAve) {
    for (auto [chamber, nNumRecHit] : mapSub) {
      auto key3 = key4Tokey3(key);
      auto ieta = keyToIEta(key);
      mapCLSNumberAve_.Fill(key3, chamber, ieta, nNumRecHit);
      mapCLSAverage_.Fill(key3, chamber, ieta, mapCLSAverage[key][chamber]);
    }
  }
  for (auto [key, mapSub] : mapCLSNumberOv5) {
    for (auto [chamber, nNumRecHit] : mapSub) {
      auto key3 = key4Tokey3(key);
      auto ieta = keyToIEta(key);
      mapCLSNumberOv5_.Fill(key3, chamber, ieta, nNumRecHit);
      mapCLSOver5_.Fill(key3, chamber, ieta, mapCLSOver5[key][chamber]);
    }
  }
}

DEFINE_FWK_MODULE(GEMRecHitSource);
