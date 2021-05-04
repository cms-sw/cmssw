#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Validation/MuonGEMHits/interface/GEMValidationUtils.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "DQM/GEM/interface/GEMDQMBase.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMDigiSource : public GEMDQMBase {
public:
  explicit GEMDigiSource(const edm::ParameterSet& cfg);
  ~GEMDigiSource() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override{};
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  int ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) override;

  edm::EDGetToken tagDigi_;

  edm::EDGetTokenT<LumiScalersCollection> lumiScalers_;

  MEMap3Inf mapTotalDigi_layer_;
  MEMap3Inf mapStripOcc_ieta_;
  MEMap3Inf mapStripOcc_phi_;
  MEMap3Inf mapTotalStripPerEvt_;
  MEMap3Inf mapBX_layer_;

  MEMap4Inf mapStripOccPerCh_;
  MEMap4Inf mapBXPerCh_;

  MonitorElement *h2SummaryOcc, *h2SummaryMal;

  Int_t nBXMin_, nBXMax_;
};

using namespace std;
using namespace edm;

GEMDigiSource::GEMDigiSource(const edm::ParameterSet& cfg) : GEMDQMBase(cfg) {
  tagDigi_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("digisInputLabel"));
  lumiScalers_ = consumes<LumiScalersCollection>(
      cfg.getUntrackedParameter<edm::InputTag>("lumiCollection", edm::InputTag("scalersRawToDigi")));
}

void GEMDigiSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digisInputLabel", edm::InputTag("muonGEMDigis", ""));
  desc.addUntracked<std::string>("logCategory", "GEMDigiSource");
  descriptions.add("GEMDigiSource", desc);
}

void GEMDigiSource::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;
  loadChambers();

  nBXMin_ = -10;
  nBXMax_ = 10;

  mapTotalDigi_layer_ = MEMap3Inf(this, "digi_det", "Digi Occupancy", 36, 0.5, 36.5, 24, 0.5, 24.5, "Chamber", "VFAT");
  mapStripOcc_ieta_ = MEMap3Inf(
      this, "strip_ieta_occ", "Strip Digi Occupancy per eta partition", 8, 0.5, 8.5, "iEta", "Number of fired strips");
  mapStripOcc_phi_ = MEMap3Inf(this,
                               "strip_phi_occ",
                               "Strip Digi Phi (degree) Occupancy ",
                               108,
                               -5,
                               355,
                               "#phi (degree)",
                               "Number of fired strips");
  mapTotalStripPerEvt_ = MEMap3Inf(this,
                                   "total_strips_per_event",
                                   "Total number of strip digis per event for each layers ",
                                   50,
                                   -0.5,
                                   99.5,
                                   "Number of fired strips",
                                   "Events");
  mapBX_layer_ =
      MEMap3Inf(this, "bx", "Strip Digi Bunch Crossing ", 21, nBXMin_ - 0.5, nBXMax_ + 0.5, "Bunch crossing");

  mapStripOccPerCh_ = MEMap4Inf(this, "strip_occ", "Strip Digi Occupancy ", 1, 0.5, 1.5, 1, 0.5, 1.5, "Strip", "iEta");
  mapBXPerCh_ = MEMap4Inf(this,
                          "bx_ch",
                          "Strip Digi Bunch Crossing ",
                          21,
                          nBXMin_ - 0.5,
                          nBXMax_ + 0.5,
                          1,
                          0.5,
                          1.5,
                          "Bunch crossing",
                          "VFAT");

  ibooker.cd();
  ibooker.setCurrentFolder("GEM/digi");
  GenerateMEPerChamber(ibooker);

  h2SummaryOcc = CreateSummaryHist(ibooker, "summaryOccDigi");
  h2SummaryMal = CreateSummaryHist(ibooker, "summaryMalfuncDigi");
}

int GEMDigiSource::ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) {
  MEStationInfo& stationInfo = mapStationInfo_[key];

  mapTotalDigi_layer_.SetBinConfX(stationInfo.nNumChambers_);
  mapTotalDigi_layer_.SetBinConfY(stationInfo.nMaxVFAT_);
  mapTotalDigi_layer_.bookND(bh, key);
  mapTotalDigi_layer_.SetLabelForChambers(key, 1);
  mapTotalDigi_layer_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  mapStripOcc_ieta_.SetBinConfX(stationInfo.nNumEtaPartitions_);
  mapStripOcc_ieta_.bookND(bh, key);
  mapTotalDigi_layer_.SetLabelForChambers(key, 1);  // For eta partitions

  mapStripOcc_phi_.bookND(bh, key);
  mapTotalStripPerEvt_.bookND(bh, key);
  mapBX_layer_.bookND(bh, key);

  return 0;
}

int GEMDigiSource::ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) {
  ME3IdsKey key3 = key4Tokey3(key);
  MEStationInfo& stationInfo = mapStationInfo_[key3];

  int nNumVFATPerRoll = stationInfo.nMaxVFAT_ / stationInfo.nNumEtaPartitions_;
  int nNumCh = stationInfo.nNumStrip_;

  mapStripOccPerCh_.SetBinConfX(nNumCh * nNumVFATPerRoll);
  mapStripOccPerCh_.SetBinConfY(stationInfo.nNumEtaPartitions_);
  mapStripOccPerCh_.bookND(bh, key);
  mapStripOccPerCh_.SetLabelForChambers(key, 2);  // For eta partitions

  mapBXPerCh_.SetBinConfY(stationInfo.nMaxVFAT_);
  mapBXPerCh_.bookND(bh, key);
  mapBXPerCh_.SetLabelForVFATs(key, stationInfo.nNumEtaPartitions_, 2);

  return 0;
}

void GEMDigiSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Handle<GEMDigiCollection> gemDigis;
  event.getByToken(this->tagDigi_, gemDigis);
  edm::Handle<LumiScalersCollection> lumiScalers;
  event.getByToken(lumiScalers_, lumiScalers);

  std::map<ME3IdsKey, Int_t> total_strip_layer;
  for (const auto& ch : gemChambers_) {
    GEMDetId gid = ch.id();
    ME2IdsKey key2{gid.region(), gid.station()};
    ME3IdsKey key3{gid.region(), gid.station(), gid.layer()};
    ME4IdsKey key4Ch{gid.region(), gid.station(), gid.layer(), gid.chamber()};
    std::map<Int_t, bool> bTagVFAT;
    bTagVFAT.clear();
    const BoundPlane& surface = GEMGeometry_->idToDet(gid)->surface();
    if (total_strip_layer.find(key3) == total_strip_layer.end())
      total_strip_layer[key3] = 0;
    for (auto roll : ch.etaPartitions()) {
      GEMDetId rId = roll->id();
      const auto& digis_in_det = gemDigis->get(rId);
      for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
        // Filling of digi occupancy
        Int_t nIdxVFAT = getVFATNumberByStrip(gid.station(), rId.roll(), d->strip());
        mapTotalDigi_layer_.Fill(key3, gid.chamber(), nIdxVFAT + 1);

        // Filling of strip
        mapStripOcc_ieta_.Fill(key3, rId.roll());  // Roll

        GlobalPoint strip_global_pos = surface.toGlobal(roll->centreOfStrip(d->strip()));
        Float_t fPhiDeg = ((Float_t)strip_global_pos.phi()) * 180.0 / 3.141592;
        if (fPhiDeg < -5.0)
          fPhiDeg += 360.0;
        mapStripOcc_phi_.Fill(key3, fPhiDeg);  // Phi

        mapStripOccPerCh_.Fill(key4Ch, d->strip(), rId.roll());  // Per chamber

        // For total strips
        total_strip_layer[key3]++;

        // Filling of bx
        Int_t nBX = std::min(std::max((Int_t)d->bx(), nBXMin_), nBXMax_);  // For under/overflow
        if (bTagVFAT.find(nIdxVFAT) == bTagVFAT.end()) {
          mapBX_layer_.Fill(key3, nBX);
          mapBXPerCh_.Fill(key4Ch, nBX, nIdxVFAT);
        }

        // Occupancy on a chamber
        h2SummaryOcc->Fill(gid.chamber(), mapStationToIdx_[key3]);

        bTagVFAT[nIdxVFAT] = true;
      }
    }
  }
  for (auto [key, num_total_strip] : total_strip_layer)
    mapTotalStripPerEvt_.Fill(key, num_total_strip);
}

DEFINE_FWK_MODULE(GEMDigiSource);
