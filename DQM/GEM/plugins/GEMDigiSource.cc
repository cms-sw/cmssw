#include "DQM/GEM/interface/GEMDigiSource.h"

using namespace std;
using namespace edm;

GEMDigiSource::GEMDigiSource(const edm::ParameterSet& cfg)
    : GEMDQMBase(cfg), gemChMapToken_(esConsumes<GEMChMap, GEMChMapRcd, edm::Transition::BeginRun>()) {
  tagDigi_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("digisInputLabel"));
  lumiScalers_ = consumes<LumiScalersCollection>(
      cfg.getUntrackedParameter<edm::InputTag>("lumiCollection", edm::InputTag("scalersRawToDigi")));
  nBXMin_ = cfg.getParameter<int>("bxMin");
  nBXMax_ = cfg.getParameter<int>("bxMax");
  useDBEMap_ = cfg.getParameter<bool>("useDBEMap");
}

void GEMDigiSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digisInputLabel", edm::InputTag("muonGEMDigis", ""));
  desc.addUntracked<std::string>("runType", "online");
  desc.addUntracked<std::string>("logCategory", "GEMDigiSource");
  desc.add<int>("bxMin", -10);
  desc.add<int>("bxMax", 10);
  desc.add<bool>("useDBEMap", true);
  descriptions.add("GEMDigiSource", desc);
}

void GEMDigiSource::LoadROMap(edm::EventSetup const& iSetup) {
  if (useDBEMap_) {
    const auto& chMap = iSetup.getData(gemChMapToken_);
    auto gemChMap = std::make_unique<GEMChMap>(chMap);

    std::map<int, bool> mapCheckedType;
    for (auto const& p : gemChMap->chamberMap()) {
      auto& dc = p.second;
      GEMDetId gemChId(dc.detId);
      for (Int_t ieta = 1; ieta <= 24; ieta++) {
        if (!gemChMap->isValidStrip(dc.chamberType, ieta, 1))
          continue;
        mapChamberType_[{gemChId.station(), gemChId.layer(), gemChId.chamber(), ieta}] = dc.chamberType;
        if (mapCheckedType[dc.chamberType])
          continue;
        for (Int_t strip = 0; strip <= 3 * 128; strip++) {
          if (!gemChMap->isValidStrip(dc.chamberType, ieta, strip))
            continue;
          auto& stripInfo = gemChMap->getChannel(dc.chamberType, ieta, strip);
          mapStripToVFAT_[{dc.chamberType, ieta, strip}] = stripInfo.vfatAdd;
        }
      }
      mapCheckedType[dc.chamberType] = true;
    }
  } else {
    // no EMap in DB, using dummy
    auto gemChMap = std::make_unique<GEMChMap>();
    gemChMap->setDummy();

    std::map<int, bool> mapCheckedType;
    for (auto const& p : gemChMap->chamberMap()) {
      auto& dc = p.second;
      GEMDetId gemChId(dc.detId);

      for (Int_t ieta = 1; ieta <= 24; ieta++) {
        if (!gemChMap->isValidStrip(dc.chamberType, ieta, 1))
          continue;
        Int_t nChamberType = dc.chamberType;
        if (gemChId.station() == 1) {
          nChamberType = 13 - gemChId.layer();
        } else if (gemChId.station() == 2) {
          nChamberType = 24 - (ieta - 1) / 4;
        }
        mapChamberType_[{gemChId.station(), gemChId.layer(), gemChId.chamber(), ieta}] = nChamberType;
        if (mapCheckedType[nChamberType])
          continue;
        mapCheckedType[nChamberType] = true;
        for (Int_t strip = 0; strip <= 3 * 128; strip++) {
          if (!gemChMap->isValidStrip(nChamberType, ieta, strip))
            continue;
          auto& stripInfo = gemChMap->getChannel(nChamberType, ieta, strip);
          mapStripToVFAT_[{nChamberType, ieta, strip}] = stripInfo.vfatAdd;
        }
      }
    }
  }
}

void GEMDigiSource::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();
  LoadROMap(iSetup);

  strFolderMain_ = "GEM/Digis";

  fRadiusMin_ = 120.0;
  fRadiusMax_ = 250.0;
  float radS = -5.0 / 180 * M_PI;
  float radL = 355.0 / 180 * M_PI;

  mapTotalDigi_layer_ = MEMap4Inf(this, "occ", "Digi Occupancy", 36, 0.5, 36.5, 24, -0.5, 24 - 0.5, "Chamber", "VFAT");
  mapDigiWheel_layer_ = MEMap3Inf(
      this, "occ_rphi", "Digi R-Phi Occupancy", 360, radS, radL, 8, fRadiusMin_, fRadiusMax_, "#phi (rad)", "R [cm]");
  mapDigiOcc_ieta_ = MEMap4Inf(this, "occ_ieta", "Digi iEta Occupancy", 8, 0.5, 8.5, "iEta", "Number of fired digis");
  mapDigiOcc_phi_ =
      MEMap4Inf(this, "occ_phi", "Digi Phi Occupancy", 72, -5, 355, "#phi (degree)", "Number of fired digis");
  mapTotalDigiPerEvtLayer_ = MEMap4Inf(this,
                                       "digis_per_layer",
                                       "Total number of digis per event for each layers",
                                       50,
                                       -0.5,
                                       99.5,
                                       "Number of fired digis",
                                       "Events");
  mapTotalDigiPerEvtLayer_.SetNoUnderOverflowBin();
  mapTotalDigiPerEvtIEta_ = MEMap3Inf(this,
                                      "digis_per_ieta",
                                      "Total number of digis per event for each eta partitions",
                                      50,
                                      -0.5,
                                      99.5,
                                      "Number of fired digis",
                                      "Events");
  mapTotalDigiPerEvtIEta_.SetNoUnderOverflowBin();

  mapBX_ = MEMap2Inf(this, "bx", "Digi Bunch Crossing", 21, nBXMin_ - 0.5, nBXMax_ + 0.5, "Bunch crossing");

  mapDigiOccPerCh_ = MEMap5Inf(this, "occ", "Digi Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Digi", "iEta");

  if (nRunType_ == GEMDQM_RUNTYPE_OFFLINE) {
    mapDigiWheel_layer_.TurnOff();
    mapBX_.TurnOff();
  }

  if (nRunType_ == GEMDQM_RUNTYPE_RELVAL) {
    mapDigiWheel_layer_.TurnOff();
    mapDigiOccPerCh_.TurnOff();
    mapTotalDigi_layer_.TurnOff();
  }

  if (nRunType_ != GEMDQM_RUNTYPE_ALLPLOTS && nRunType_ != GEMDQM_RUNTYPE_RELVAL) {
    mapDigiOcc_ieta_.TurnOff();
    mapDigiOcc_phi_.TurnOff();
  }

  if (nRunType_ != GEMDQM_RUNTYPE_ALLPLOTS) {
    mapTotalDigiPerEvtLayer_.TurnOff();
    mapTotalDigiPerEvtIEta_.TurnOff();
  }

  ibooker.cd();
  ibooker.setCurrentFolder(strFolderMain_);
  GenerateMEPerChamber(ibooker);
}

int GEMDigiSource::ProcessWithMEMap2(BookingHelper& bh, ME2IdsKey key) {
  mapBX_.bookND(bh, key);

  return 0;
}

int GEMDigiSource::ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) {
  mapTotalDigiPerEvtIEta_.bookND(bh, key);

  return 0;
}

int GEMDigiSource::ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) {
  MEStationInfo& stationInfo = mapStationInfo_[key];

  int nNumVFATPerEta = stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_;

  mapDigiWheel_layer_.SetBinLowEdgeX(stationInfo.fMinPhi_);
  mapDigiWheel_layer_.SetBinHighEdgeX(stationInfo.fMinPhi_ + 2 * M_PI);
  mapDigiWheel_layer_.SetNbinsX(nNumVFATPerEta * stationInfo.nNumChambers_);
  mapDigiWheel_layer_.SetNbinsY(stationInfo.nNumEtaPartitions_);
  mapDigiWheel_layer_.bookND(bh, key);

  return 0;
}

int GEMDigiSource::ProcessWithMEMap4(BookingHelper& bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  Int_t nNewNumCh = stationInfo.nMaxIdxChamber_ - stationInfo.nMinIdxChamber_ + 1;
  Int_t nNewMinIdxChamber = (stationInfo.nMinIdxChamber_ - 1) + 1;
  Int_t nNewMaxIdxChamber = stationInfo.nMaxIdxChamber_;

  Int_t nNumVFATPerModule = stationInfo.nMaxVFAT_ / stationInfo.nNumModules_;

  mapTotalDigi_layer_.SetBinConfX(nNewNumCh, nNewMinIdxChamber - 0.5, nNewMaxIdxChamber + 0.5);
  mapTotalDigi_layer_.SetBinConfY(nNumVFATPerModule, -0.5);
  mapTotalDigi_layer_.bookND(bh, key);
  mapTotalDigi_layer_.SetLabelForChambers(key, 1, -1, nNewMinIdxChamber);
  mapTotalDigi_layer_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  mapDigiOcc_ieta_.SetBinConfX(stationInfo.nNumEtaPartitions_);
  mapDigiOcc_ieta_.bookND(bh, key);
  mapDigiOcc_ieta_.SetLabelForIEta(key, 1);

  mapDigiOcc_phi_.SetBinLowEdgeX(stationInfo.fMinPhi_ * 180 / M_PI);
  mapDigiOcc_phi_.SetBinHighEdgeX(stationInfo.fMinPhi_ * 180 / M_PI + 360);
  mapDigiOcc_phi_.bookND(bh, key);
  mapTotalDigiPerEvtLayer_.bookND(bh, key);

  return 0;
}

int GEMDigiSource::ProcessWithMEMap5WithChamber(BookingHelper& bh, ME5IdsKey key) {
  ME4IdsKey key4 = key5Tokey4(key);
  ME3IdsKey key3 = key4Tokey3(key4);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  bh.getBooker()->setCurrentFolder(strFolderMain_ + "/occupancy_" + getNameDirLayer(key4));

  int nNumVFATPerEta = stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_;
  int nNumCh = stationInfo.nNumDigi_;

  mapDigiOccPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta, stationInfo.nFirstStrip_ - 0.5);
  mapDigiOccPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_ / stationInfo.nNumModules_);
  mapDigiOccPerCh_.bookND(bh, key);
  mapDigiOccPerCh_.SetLabelForIEta(key, 2);

  bh.getBooker()->setCurrentFolder(strFolderMain_);

  return 0;
}

void GEMDigiSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<GEMDigiCollection> gemDigis;
  event.getByToken(this->tagDigi_, gemDigis);
  edm::Handle<LumiScalersCollection> lumiScalers;
  event.getByToken(lumiScalers_, lumiScalers);

  std::map<ME4IdsKey, Int_t> total_digi_layer;
  std::map<ME3IdsKey, Int_t> total_digi_eta;
  for (auto gid : listChamberId_) {
    ME2IdsKey key2{gid.region(), gid.station()};
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
    std::map<Int_t, bool> bTagVFAT;
    bTagVFAT.clear();
    MEStationInfo& stationInfo = mapStationInfo_[key3];
    const BoundPlane& surface = GEMGeometry_->idToDet(gid)->surface();
    for (auto iEta : mapEtaPartition_[gid]) {
      GEMDetId eId = iEta->id();
      ME3IdsKey key3IEta{gid.region(), gid.station(), eId.ieta()};
      if (total_digi_eta.find(key3IEta) == total_digi_eta.end())
        total_digi_eta[key3IEta] = 0;
      const auto& digis_in_det = gemDigis->get(eId);
      auto nChamberType = mapChamberType_[{gid.station(), gid.layer(), gid.chamber(), eId.ieta()}];
      Int_t nIdxModule = getIdxModule(gid.station(), nChamberType);
      ME4IdsKey key4{gid.region(), gid.station(), gid.layer(), nIdxModule};
      ME5IdsKey key5Ch{gid.region(), gid.station(), gid.layer(), nIdxModule, gid.chamber()};
      if (total_digi_layer.find(key4) == total_digi_layer.end())
        total_digi_layer[key4] = 0;
      for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
        // Filling of digi occupancy
        Int_t nIdxVFAT = mapStripToVFAT_[{nChamberType, eId.ieta(), d->strip()}];
        mapTotalDigi_layer_.Fill(key4, gid.chamber(), nIdxVFAT);

        // Filling of digi
        mapDigiOcc_ieta_.Fill(key4, eId.ieta());  // Eta (partition)

        GlobalPoint digi_global_pos = surface.toGlobal(iEta->centreOfStrip(d->strip()));
        Float_t fPhi = (Float_t)digi_global_pos.phi();
        Float_t fPhiShift = restrictAngle(fPhi, stationInfo.fMinPhi_);
        Float_t fPhiDeg = fPhiShift * 180.0 / M_PI;
        mapDigiOcc_phi_.Fill(key4, fPhiDeg);  // Phi

        // Filling of R-Phi occupancy
        Float_t fR = fRadiusMin_ + (fRadiusMax_ - fRadiusMin_) * (eId.ieta() - 0.5) / stationInfo.nNumEtaPartitions_;
        mapDigiWheel_layer_.Fill(key3, fPhiShift, fR);

        Int_t nIEtaInMod = eId.ieta();
        if (gid.station() == 2) {
          nIEtaInMod = (eId.ieta() - 1) % stationInfo.nNumModules_ + 1;
        }
        mapDigiOccPerCh_.Fill(key5Ch, d->strip(), nIEtaInMod);  // Per chamber

        // For total digis
        total_digi_layer[key4]++;
        total_digi_eta[key3IEta]++;

        // Filling of bx
        Int_t nBX = std::min(std::max((Int_t)d->bx(), nBXMin_), nBXMax_);  // For under/overflow
        if (bTagVFAT.find(nIdxVFAT) == bTagVFAT.end()) {
          mapBX_.Fill(key2, nBX);
        }

        bTagVFAT[nIdxVFAT] = true;
      }
    }
  }
  for (auto [key, num_total_digi] : total_digi_layer)
    mapTotalDigiPerEvtLayer_.Fill(key, num_total_digi);
  for (auto [key, num_total_digi] : total_digi_eta)
    mapTotalDigiPerEvtIEta_.Fill(key, num_total_digi);
}

DEFINE_FWK_MODULE(GEMDigiSource);
