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

  typedef std::map<ME4IdsKey, std::map<Int_t, Int_t>> ME5map;

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override{};
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  int ProcessWithMEMap2WithEta(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap2AbsReWithEta(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap3(BookingHelper& bh, ME3IdsKey key) override;
  int ProcessWithMEMap4(BookingHelper& bh, ME4IdsKey key) override;
  int ProcessWithMEMap3WithChamber(BookingHelper& bh, ME4IdsKey key) override;
  
  void InitKey5MultiMap(ME5map &map, ME4IdsKey key1, Int_t key2) {
    if (map.find(key1) == map.end()) map[key1][key2] = 0;
    else if (map[key1].find(key2) == map[key1].end()) map[key1][key2] = 0;
  };

  edm::EDGetToken tagRecHit_;

  float fGlobXMin_, fGlobXMax_;
  float fGlobYMin_, fGlobYMax_;

  int nIdxFirstStrip_;
  int nClusterSizeBinNum_;

  MEMap3Inf mapTotalRecHit_layer_;
  MEMap3Inf mapRecHitWheel_layer_;
  MEMap3Inf mapRecHitOcc_ieta_;
  MEMap3Inf mapRecHitOcc_phi_;
  MEMap3Inf mapTotalRecHitPerEvtLayer_;
  MEMap3Inf mapTotalRecHitPerEvtIEta_;
  //MEMap4Inf mapCLSRecHit_ieta_;
  MEMap3Inf mapCLSRecHit_ieta_;
  MEMap3Inf mapCLSNumber_;
  MEMap3Inf mapCLSAverage_;
  MEMap3Inf mapCLSOver5_;

  MEMap4Inf mapCLSPerCh_;

  Int_t nCLSMax_;
  Float_t fRadiusMin_;
  Float_t fRadiusMax_;

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
  ibooker.setCurrentFolder("GEM/RecHits");

  nCLSMax_ = 10;
  fRadiusMin_ = 120.0;
  fRadiusMax_ = 250.0;
  float radS = -5.0 / 180 * 3.141592;
  float radL = 355.0 / 180 * 3.141592;

  mapTotalRecHit_layer_ =
      MEMap3Inf(this, "rechit_det", "Rec. Hit Occupancy", 36, 0.5, 36.5, 8, 0.5, 8.5, "Chamber", "iEta");
  mapRecHitWheel_layer_ =
      MEMap3Inf(this, "rechit_wheel", "Rec. Hit Wheel Occupancy", 108, radS, radL, 8, fRadiusMin_, fRadiusMax_, "", "");
  mapRecHitOcc_ieta_ = MEMap3Inf(
      this, "rechit_ieta_occ", "RecHit Digi Occupancy per eta partition", 8, 0.5, 8.5, "iEta", "Number of recHits");
  mapRecHitOcc_phi_ = MEMap3Inf(
      this, "rechit_phi_occ", "RecHit Digi Phi (degree) Occupancy ", 108, -5, 355, "#phi (degree)", "Number of recHits");
  mapTotalRecHitPerEvtLayer_ = MEMap3Inf(this,
                                         "total_rechit_per_event",
                                         "Total number of rec. hits per event for each layers ",
                                         50,
                                         -0.5,
                                         99.5,
                                         "Number of recHits",
                                         "Events");
  mapTotalRecHitPerEvtIEta_ = MEMap3Inf(this,
                                        "total_rechit_per_event",
                                        "Total number of rec. hits per event for each eta partitions ",
                                        50,
                                        -0.5,
                                        99.5,
                                        "Number of recHits",
                                        "Events");
  //mapCLSRecHit_ieta_ = MEMap4Inf(
  //    this, "cls", "Cluster size of rec. hits", nCLSMax_, 0.5, nCLSMax_ + 0.5, "Cluster size", "Number of recHits");
  mapCLSRecHit_ieta_ = MEMap3Inf(
      this, "cls", "Cluster size of rec. hits", nCLSMax_, 0.5, nCLSMax_ + 0.5, "Cluster size", "Number of recHits");
  mapCLSNumber_  = MEMap3Inf(this, "rechitNumber", "", 36, 0.5, 36.5, 8, 0.5, 8.5, "Chamber", "iEta");
  mapCLSAverage_ = MEMap3Inf(this, "rechit_average_pre", "", 36, 0.5, 36.5, 8, 0.5, 8.5, "Chamber", "iEta");
  mapCLSOver5_   = MEMap3Inf(this, "rechit_over5_pre", "", 36, 0.5, 36.5, 8, 0.5, 8.5, "Chamber", "iEta");

  mapCLSPerCh_ = MEMap4Inf(
      this, "cls", "Cluster size of rec. hits", nCLSMax_, 0.5, nCLSMax_ + 0.5, 1, 0.5, 1.5, "Cluster size", "iEta");

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

  mapCLSNumber_.bookND(bh, key);
  mapCLSAverage_.bookND(bh, key);
  mapCLSOver5_.bookND(bh, key);
  mapCLSAverage_.SetLabelForChambers(key, 1);
  mapCLSAverage_.SetLabelForIEta(key, 2);
  mapCLSOver5_.SetLabelForChambers(key, 1);
  mapCLSOver5_.SetLabelForIEta(key, 2);

  return 0;
}

int GEMRecHitSource::ProcessWithMEMap4(BookingHelper& bh, ME4IdsKey key) {
  //mapCLSRecHit_ieta_.bookND(bh, key);

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
    edm::LogError(log_category_) << "GEM recHit is not valid.\n";
    return;
  }

  std::map<ME3IdsKey, Int_t> total_rechit_layer;
  std::map<ME3IdsKey, Int_t> total_rechit_iEta;
  ME5map mapCLSNumber;
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
      InitKey5MultiMap(mapCLSNumber,  key4IEta, chamber);
      InitKey5MultiMap(mapCLSAverage, key4IEta, chamber);
      InitKey5MultiMap(mapCLSOver5,   key4IEta, chamber);

      const auto& recHitsRange = gemRecHits->get(eId);
      auto gemRecHit = recHitsRange.first;
      Int_t nNumRecHit = 0;
      for (auto hit = gemRecHit; hit != recHitsRange.second; ++hit) {
        GlobalPoint recHitGP = GEMGeometry_->idToDet(hit->gemId())->surface().toGlobal(hit->localPosition());
        Float_t fPhi = recHitGP.phi();
        if (fPhi < -5.0 / 180.0 * 3.141592)
          fPhi += 2 * 3.141592;

        // Filling of recHit occupancy
        mapTotalRecHit_layer_.Fill(key3, chamber, eId.ieta());

        // Filling of wheel occupancy
        Float_t fR = fRadiusMin_ + (fRadiusMax_ - fRadiusMin_) * (eId.ieta() - 0.5) / stationInfo.nNumEtaPartitions_;
        mapRecHitWheel_layer_.Fill(key3, fPhi, fR);

        // Filling of strip (iEta)
        mapRecHitOcc_ieta_.Fill(key3, eId.ieta());

        // Filling of strip (phi)
        Float_t fPhiDeg = fPhi * 180.0 / 3.141592;
        mapRecHitOcc_phi_.Fill(key3, fPhiDeg);

        // For total recHits
        total_rechit_layer[key3]++;
        total_rechit_iEta[key3IEta]++;

        // Filling of cluster size (CLS)
        Int_t nCLS = hit->clusterSize();
        Int_t nCLSCutOff = std::min(nCLS, nCLSMax_);  // For overflow
        //mapCLSRecHit_ieta_.Fill(key4IEta, nCLS);
        mapCLSRecHit_ieta_.Fill(key3AbsReIEta, nCLSCutOff);
        mapCLSPerCh_.Fill(key4Ch, eId.ieta(), nCLSCutOff);
        mapCLSAverage[key4IEta][chamber] += nCLS;
        if ( nCLS > 5 ) mapCLSOver5[key4IEta][chamber] += nCLS;
        nNumRecHit++;
      }
      mapCLSNumber[key4IEta][chamber] = nNumRecHit;
    }
  }
  for (auto [key, num_total_rechit] : total_rechit_layer) {
    mapTotalRecHitPerEvtLayer_.Fill(key, num_total_rechit);
  }
  for (auto [key, num_total_rechit] : total_rechit_iEta) {
    mapTotalRecHitPerEvtIEta_.Fill(key, num_total_rechit);
  }
  for (auto [key, mapSub] : mapCLSNumber) {
    for (auto [chamber, nNumRecHit] : mapSub) {
      auto key3 = key4Tokey3(key);
      auto ieta = keyToIEta(key);
      mapCLSNumber_.Fill(key3,  chamber, ieta, nNumRecHit);
      mapCLSAverage_.Fill(key3, chamber, ieta, mapCLSAverage[key][chamber]);
      mapCLSOver5_.Fill(key3,   chamber, ieta, mapCLSOver5[key][chamber]);
    }
  }
}

DEFINE_FWK_MODULE(GEMRecHitSource);
