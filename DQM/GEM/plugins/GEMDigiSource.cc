#include "DQM/GEM/interface/GEMDigiSource.h"

using namespace std;
using namespace edm;

GEMDigiSource::GEMDigiSource(const edm::ParameterSet& cfg) : GEMDQMBase(cfg) {
  tagDigi_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("digisInputLabel"));
  lumiScalers_ = consumes<LumiScalersCollection>(
      cfg.getUntrackedParameter<edm::InputTag>("lumiCollection", edm::InputTag("scalersRawToDigi")));
  bModeRelVal_ = cfg.getParameter<bool>("modeRelVal");
}

void GEMDigiSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digisInputLabel", edm::InputTag("muonGEMDigis", ""));
  desc.addUntracked<std::string>("logCategory", "GEMDigiSource");
  desc.add<bool>("modeRelVal", false);
  descriptions.add("GEMDigiSource", desc);
}

void GEMDigiSource::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();

  nBXMin_ = -10;
  nBXMax_ = 10;

  mapTotalDigi_layer_ = MEMap3Inf(this, "det", "Digi Occupancy", 36, 0.5, 36.5, 24, -0.5, 24 - 0.5, "Chamber", "VFAT");
  mapDigiOcc_ieta_ = MEMap3Inf(this, "occ_ieta", "Digi iEta Occupancy", 8, 0.5, 8.5, "iEta", "Number of fired digis");
  mapDigiOcc_phi_ =
      MEMap3Inf(this, "occ_phi", "Digi Phi Occupancy", 108, -5, 355, "#phi (degree)", "Number of fired digis");
  mapTotalDigiPerEvtLayer_ = MEMap3Inf(this,
                                       "digis_per_layer",
                                       "Total number of digis per event for each layers",
                                       50,
                                       -0.5,
                                       99.5,
                                       "Number of fired digis",
                                       "Events");
  mapTotalDigiPerEvtIEta_ = MEMap3Inf(this,
                                      "digis_per_ieta",
                                      "Total number of digis per event for each eta partitions",
                                      50,
                                      -0.5,
                                      99.5,
                                      "Number of fired digis",
                                      "Events");

  mapBX_iEta_ = MEMap3Inf(this, "bx", "Digi Bunch Crossing", 21, nBXMin_ - 0.5, nBXMax_ + 0.5, "Bunch crossing");

  mapDigiOccPerCh_ = MEMap4Inf(this, "occ", "Digi Occupancy", 1, -0.5, 1.5, 1, 0.5, 1.5, "Digi", "iEta");

  if (bModeRelVal_) {
    mapTotalDigi_layer_.TurnOff();
    mapDigiOccPerCh_.TurnOff();
  }

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/Digis");
  GenerateMEPerChamber(ibooker);

  h2SummaryOcc_ = nullptr;
  if (!bModeRelVal_) {
    h2SummaryOcc_ = CreateSummaryHist(ibooker, "summaryOccDigi");
    h2SummaryOcc_->setTitle("Summary of occupancy on chambers");
    h2SummaryOcc_->setXTitle("Chamber");
  }
}

int GEMDigiSource::ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) {
  mapBX_iEta_.bookND(bh, key);
  mapTotalDigiPerEvtIEta_.bookND(bh, key);

  return 0;
}

int GEMDigiSource::ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) {
  MEStationInfo& stationInfo = mapStationInfo_[key];

  mapTotalDigi_layer_.SetBinConfX(stationInfo.nNumChambers_);
  mapTotalDigi_layer_.SetBinConfY(stationInfo.nMaxVFAT_, -0.5);
  mapTotalDigi_layer_.bookND(bh, key);
  mapTotalDigi_layer_.SetLabelForChambers(key, 1);
  mapTotalDigi_layer_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  mapDigiOcc_ieta_.SetBinConfX(stationInfo.nNumEtaPartitions_);
  mapDigiOcc_ieta_.bookND(bh, key);
  mapDigiOcc_ieta_.SetLabelForIEta(key, 1);

  mapDigiOcc_phi_.bookND(bh, key);
  mapTotalDigiPerEvtLayer_.bookND(bh, key);

  return 0;
}

int GEMDigiSource::ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  int nNumVFATPerEta = stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_;
  int nNumCh = stationInfo.nNumDigi_;

  mapDigiOccPerCh_.SetBinConfX(nNumCh * nNumVFATPerEta);
  mapDigiOccPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapDigiOccPerCh_.bookND(bh, key);
  mapDigiOccPerCh_.SetLabelForIEta(key, 2);

  return 0;
}

void GEMDigiSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<GEMDigiCollection> gemDigis;
  event.getByToken(this->tagDigi_, gemDigis);
  edm::Handle<LumiScalersCollection> lumiScalers;
  event.getByToken(lumiScalers_, lumiScalers);

  std::map<ME3IdsKey, Int_t> total_digi_layer;
  std::map<ME3IdsKey, Int_t> total_digi_eta;
  for (const auto& ch : gemChambers_) {
    GEMDetId gid = ch.id();
    ME2IdsKey key2{gid.region(), gid.station()};
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
    std::map<Int_t, bool> bTagVFAT;
    bTagVFAT.clear();
    const BoundPlane& surface = GEMGeometry_->idToDet(gid)->surface();
    if (total_digi_layer.find(key3) == total_digi_layer.end())
      total_digi_layer[key3] = 0;
    for (auto iEta : ch.etaPartitions()) {
      GEMDetId eId = iEta->id();
      ME3IdsKey key3IEta{gid.region(), gid.station(), eId.ieta()};
      if (total_digi_eta.find(key3IEta) == total_digi_eta.end())
        total_digi_eta[key3IEta] = 0;
      const auto& digis_in_det = gemDigis->get(eId);
      for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
        // Filling of digi occupancy
        Int_t nIdxVFAT = getVFATNumberByDigi(gid.station(), eId.ieta(), d->strip());
        mapTotalDigi_layer_.Fill(key3, gid.chamber(), nIdxVFAT);

        // Filling of digi
        mapDigiOcc_ieta_.Fill(key3, eId.ieta());  // Eta (partition)

        GlobalPoint digi_global_pos = surface.toGlobal(iEta->centreOfStrip(d->strip()));
        Float_t fPhiDeg = ((Float_t)digi_global_pos.phi()) * 180.0 / 3.141592;
        if (fPhiDeg < -5.0)
          fPhiDeg += 360.0;
        mapDigiOcc_phi_.Fill(key3, fPhiDeg);  // Phi

        mapDigiOccPerCh_.Fill(key4Ch, d->strip(), eId.ieta());  // Per chamber

        // For total digis
        total_digi_layer[key3]++;
        total_digi_eta[key3IEta]++;

        // Filling of bx
        Int_t nBX = std::min(std::max((Int_t)d->bx(), nBXMin_), nBXMax_);  // For under/overflow
        if (bTagVFAT.find(nIdxVFAT) == bTagVFAT.end()) {
          mapBX_iEta_.Fill(key3IEta, nBX);
        }

        // Occupancy on a chamber
        if (h2SummaryOcc_)
          h2SummaryOcc_->Fill(gid.chamber(), mapStationToIdx_[key3]);

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
