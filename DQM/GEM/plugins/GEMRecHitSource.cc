#include "DQM/GEM/interface/GEMRecHitSource.h"

using namespace std;
using namespace edm;

GEMRecHitSource::GEMRecHitSource(const edm::ParameterSet& cfg) : GEMDQMBase(cfg) {
  tagRecHit_ = consumes<GEMRecHitCollection>(cfg.getParameter<edm::InputTag>("recHitsInputLabel"));

  nIdxFirstDigi_ = cfg.getParameter<int>("idxFirstDigi");
  nNumDivideEtaPartitionInRPhi_ = cfg.getParameter<int>("numDivideEtaPartitionInRPhi");
  nCLSMax_ = cfg.getParameter<int>("clsMax");
  nClusterSizeBinNum_ = cfg.getParameter<int>("ClusterSizeBinNum");
}

void GEMRecHitSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHitsInputLabel", edm::InputTag("gemRecHits", ""));
  desc.addUntracked<std::string>("runType", "online");

  desc.add<int>("idxFirstDigi", 0);
  desc.add<int>("numDivideEtaPartitionInRPhi", 10);
  desc.add<int>("clsMax", 10);
  desc.add<int>("ClusterSizeBinNum", 9);

  desc.addUntracked<std::string>("logCategory", "GEMRecHitSource");

  descriptions.add("GEMRecHitSource", desc);
}

void GEMRecHitSource::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  std::vector<GEMDetId> listLayerOcc;

  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();

  strFolderMain_ = "GEM/RecHits";

  ibooker.cd();
  ibooker.setCurrentFolder(strFolderMain_);

  mapRecHitXY_layer_ =
      MEMap3Inf(this, "occ_xy", "RecHit xy Occupancy", 160, -250, 250, 160, -250, 250, "X [cm]", "Y [cm]");
  mapRecHitOcc_ieta_ = MEMap3Inf(this, "occ_ieta", "RecHit iEta Occupancy", 8, 0.5, 8.5, "iEta", "Number of RecHits");
  mapRecHitOcc_phi_ =
      MEMap3Inf(this, "occ_phi", "RecHit Phi Occupancy", 72, -5, 355, "#phi (degree)", "Number of RecHits");
  mapTotalRecHitPerEvtLayer_ = MEMap3Inf(this,
                                         "rechits_per_layer",
                                         "Total number of RecHits per event for each layers",
                                         2000,
                                         -0.5,
                                         2000 - 0.5,
                                         "Number of RecHits",
                                         "Events");
  mapTotalRecHitPerEvtLayer_.SetNoUnderOverflowBin();
  mapTotalRecHitPerEvtIEta_ = MEMap3Inf(this,
                                        "rechits_per_ieta",
                                        "Total number of RecHits per event for each eta partitions",
                                        300,
                                        -0.5,
                                        300 - 0.5,
                                        "Number of RecHits",
                                        "Events");
  mapTotalRecHitPerEvtIEta_.SetNoUnderOverflowBin();
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

  if (nRunType_ == GEMDQM_RUNTYPE_OFFLINE) {
    mapCLSOver5_.TurnOff();
    mapCLSPerCh_.TurnOff();
  }

  if (nRunType_ == GEMDQM_RUNTYPE_RELVAL) {
    mapRecHitXY_layer_.TurnOff();
    mapCLSAverage_.TurnOff();
    mapCLSOver5_.TurnOff();
    mapCLSPerCh_.TurnOff();
  }

  if (nRunType_ != GEMDQM_RUNTYPE_ALLPLOTS && nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
    mapRecHitOcc_ieta_.TurnOff();
    mapRecHitOcc_phi_.TurnOff();
    mapCLSRecHit_ieta_.TurnOff();
  }

  if (nRunType_ != GEMDQM_RUNTYPE_ALLPLOTS) {
    mapTotalRecHitPerEvtLayer_.TurnOff();
    mapTotalRecHitPerEvtIEta_.TurnOff();
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

  Float_t fR1 = !stationInfo.listRadiusEvenChamber_.empty() ? stationInfo.listRadiusEvenChamber_.back() : 0.0;
  Float_t fR2 = !stationInfo.listRadiusOddChamber_.empty() ? stationInfo.listRadiusOddChamber_.back() : 0.0;
  Float_t fRangeRadius = (int)(std::max(fR1, fR2) * 0.11 + 0.99999) * 10.0;

  mapRecHitXY_layer_.SetBinLowEdgeX(-fRangeRadius);
  mapRecHitXY_layer_.SetBinHighEdgeX(fRangeRadius);
  mapRecHitXY_layer_.SetBinLowEdgeY(-fRangeRadius);
  mapRecHitXY_layer_.SetBinHighEdgeY(fRangeRadius);
  mapRecHitXY_layer_.bookND(bh, key);

  mapRecHitOcc_ieta_.SetBinConfX(stationInfo.nNumEtaPartitions_);
  mapRecHitOcc_ieta_.bookND(bh, key);
  mapRecHitOcc_ieta_.SetLabelForIEta(key, 1);

  mapRecHitOcc_phi_.SetBinLowEdgeX(stationInfo.fMinPhi_ * 180 / M_PI);
  mapRecHitOcc_phi_.SetBinHighEdgeX(stationInfo.fMinPhi_ * 180 / M_PI + 360);
  mapRecHitOcc_phi_.bookND(bh, key);

  mapTotalRecHitPerEvtLayer_.bookND(bh, key);

  Int_t nNewNumCh = stationInfo.nMaxIdxChamber_ - stationInfo.nMinIdxChamber_ + 1;

  mapCLSAverage_.SetBinConfX(nNewNumCh, stationInfo.nMinIdxChamber_ - 0.5, stationInfo.nMaxIdxChamber_ + 0.5);
  mapCLSAverage_.bookND(bh, key);
  mapCLSAverage_.SetLabelForChambers(key, 1, -1, stationInfo.nMinIdxChamber_);
  mapCLSAverage_.SetLabelForIEta(key, 2);

  mapCLSOver5_.SetBinConfX(nNewNumCh, stationInfo.nMinIdxChamber_ - 0.5, stationInfo.nMaxIdxChamber_ + 0.5);
  mapCLSOver5_.bookND(bh, key);
  mapCLSOver5_.SetLabelForChambers(key, 1, -1, stationInfo.nMinIdxChamber_);
  mapCLSOver5_.SetLabelForIEta(key, 2);

  return 0;
}

int GEMRecHitSource::ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/clusterSize_" + getNameDirLayer(key3));

  mapCLSPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapCLSPerCh_.bookND(bh, key);
  mapCLSPerCh_.SetLabelForIEta(key, 2);

  bh.getBooker()->setCurrentFolder(strFolderMain_);

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
    for (auto iEta : ch.etaPartitions()) {
      GEMDetId eId = iEta->id();
      ME3IdsKey key3IEta{gid.region(), gid.station(), eId.ieta()};
      ME3IdsKey key3AbsReIEta{std::abs(gid.region()), gid.station(), eId.ieta()};
      ME4IdsKey key4IEta{gid.region(), gid.station(), gid.layer(), eId.ieta()};

      if (total_rechit_layer.find(key3) == total_rechit_layer.end())
        total_rechit_layer[key3] = 0;

      const auto& recHitsRange = gemRecHits->get(eId);
      auto gemRecHit = recHitsRange.first;
      for (auto hit = gemRecHit; hit != recHitsRange.second; ++hit) {
        LocalPoint recHitLP = hit->localPosition();
        GlobalPoint recHitGP = GEMGeometry_->idToDet(hit->gemId())->surface().toGlobal(recHitLP);

        // Filling of XY occupancy
        mapRecHitXY_layer_.Fill(key3, recHitGP.x(), recHitGP.y());

        // Filling of RecHit (iEta)
        mapRecHitOcc_ieta_.Fill(key3, eId.ieta());

        // Filling of RecHit (phi)
        Float_t fPhi = recHitGP.phi();
        Float_t fPhiShift = restrictAngle(fPhi, stationInfo.fMinPhi_);
        Float_t fPhiDeg = fPhiShift * 180.0 / M_PI;
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
