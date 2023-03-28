/****************************************************************************
 *
 * This is a part of TotemDQM and TOTEM offline software.
 * Authors:
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/OnlineMetaData/interface/CTPPSRecord.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSCommonDQMSource : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<std::vector<int>>> {
public:
  CTPPSCommonDQMSource(const edm::ParameterSet &ps);
  ~CTPPSCommonDQMSource() override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;
  std::shared_ptr<std::vector<int>> globalBeginLuminosityBlock(const edm::LuminosityBlock &iLumi,
                                                               const edm::EventSetup &c) const override;
  void globalEndLuminosityBlock(const edm::LuminosityBlock &iLumi, const edm::EventSetup &c) override;

  void analyzeCTPPSRecord(edm::Event const &event, edm::EventSetup const &eventSetup);
  void analyzeTracks(edm::Event const &event, edm::EventSetup const &eventSetup);
  void analyzeProtons(edm::Event const &event, edm::EventSetup const &eventSetup);

private:
  const unsigned int verbosity;
  constexpr static int MAX_LUMIS = 6000;
  constexpr static int MAX_VBINS = 18;

  const edm::EDGetTokenT<CTPPSRecord> ctppsRecordToken;
  const edm::EDGetTokenT<std::vector<CTPPSLocalTrackLite>> tokenLocalTrackLite;
  const edm::EDGetTokenT<std::vector<reco::ForwardProton>> tokenRecoProtons;

  bool makeProtonRecoPlots_;
  bool perLSsaving_;  //to avoid nanoDQMIO crashing, driven by  DQMServices/Core/python/DQMStore_cfi.py

  int currentLS;
  int endLS;

  std::vector<int> rpstate;

  /// plots related to the whole system
  struct GlobalPlots {
    MonitorElement *RPState = nullptr;
    MonitorElement *events_per_bx = nullptr, *events_per_bx_short = nullptr;
    MonitorElement *h_trackCorr_hor = nullptr, *h_trackCorr_vert = nullptr;

    void Init(DQMStore::IBooker &ibooker);
  };

  GlobalPlots globalPlots;

  /// plots related to one arm
  struct ArmPlots {
    int id;

    MonitorElement *h_numRPWithTrack_top = nullptr, *h_numRPWithTrack_hor = nullptr, *h_numRPWithTrack_bot = nullptr;
    MonitorElement *h_trackCorr = nullptr, *h_trackCorr_overlap = nullptr;

    MonitorElement *h_proton_xi = nullptr, *h_proton_th_x = nullptr, *h_proton_th_y = nullptr, *h_proton_t = nullptr,
                   *h_proton_time = nullptr;

    struct TrackingRPPlots {
      MonitorElement *h_x, *h_y;
    };

    std::map<unsigned int, TrackingRPPlots> trackingRPPlots;

    struct TimingRPPlots {
      MonitorElement *h_x, *h_time;
    };

    std::map<unsigned int, TimingRPPlots> timingRPPlots;

    ArmPlots() {}

    ArmPlots(DQMStore::IBooker &ibooker, int _id, bool makeProtonRecoPlots);
  };

  std::map<unsigned int, ArmPlots> armPlots;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

const int CTPPSCommonDQMSource::MAX_LUMIS;
const int CTPPSCommonDQMSource::MAX_VBINS;

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::GlobalPlots::Init(DQMStore::IBooker &ibooker) {
  ibooker.setCurrentFolder("CTPPS/common");

  events_per_bx = ibooker.book1D("events per BX", "rp;Event.BX", 4002, -1.5, 4000. + 0.5);
  events_per_bx_short = ibooker.book1D("events per BX (short)", "rp;Event.BX", 102, -1.5, 100. + 0.5);

  /*
     RP State (HV & LV & Insertion):
     0 -> not used
     1 -> bad
     2 -> warning
     3 -> ok
  */
  RPState = ibooker.book2D("rpstate per LS",
                           "RP State per Lumisection;Luminosity Section;",
                           MAX_LUMIS,
                           0,
                           MAX_LUMIS,
                           MAX_VBINS,
                           0.,
                           MAX_VBINS);
  {
    TH2F *hist = RPState->getTH2F();
    hist->SetCanExtend(TH1::kAllAxes);
    TAxis *ya = hist->GetYaxis();
    ya->SetBinLabel(1, "45, 210, FR-BT");
    ya->SetBinLabel(2, "45, 210, FR-HR");
    ya->SetBinLabel(3, "45, 210, FR-TP");
    ya->SetBinLabel(4, "45, 220, C1");
    ya->SetBinLabel(5, "45, 220, FR-BT");
    ya->SetBinLabel(6, "45, 220, FR-HR");
    ya->SetBinLabel(7, "45, 220, FR-TP");
    ya->SetBinLabel(8, "45, 220, NR-BP");
    ya->SetBinLabel(9, "45, 220, NR-TP");
    ya->SetBinLabel(10, "56, 210, FR-BT");
    ya->SetBinLabel(11, "56, 210, FR-HR");
    ya->SetBinLabel(12, "56, 210, FR-TP");
    ya->SetBinLabel(13, "56, 220, C1");
    ya->SetBinLabel(14, "56, 220, FR-BT");
    ya->SetBinLabel(15, "56, 220, FR-HR");
    ya->SetBinLabel(16, "56, 220, FR-TP");
    ya->SetBinLabel(17, "56, 220, NR-BP");
    ya->SetBinLabel(18, "56, 220, NR-TP");
  }

  h_trackCorr_hor = ibooker.book2D("track correlation hor", "ctpps_common_rp_hor", 8, -0.5, 7.5, 8, -0.5, 7.5);
  {
    TH2F *hist = h_trackCorr_hor->getTH2F();
    TAxis *xa = hist->GetXaxis(), *ya = hist->GetYaxis();
    xa->SetBinLabel(1, "45, 210, FR");
    ya->SetBinLabel(1, "45, 210, FR");
    xa->SetBinLabel(2, "45, 220, NR");
    ya->SetBinLabel(2, "45, 220, NR");
    xa->SetBinLabel(3, "45, 220, C1");
    ya->SetBinLabel(3, "45, 220, C1");
    xa->SetBinLabel(4, "45, 220, FR");
    ya->SetBinLabel(4, "45, 220, FR");

    xa->SetBinLabel(5, "56, 210, FR");
    ya->SetBinLabel(5, "56, 210, FR");
    xa->SetBinLabel(6, "56, 220, NR");
    ya->SetBinLabel(6, "56, 220, NR");
    xa->SetBinLabel(7, "56, 220, C1");
    ya->SetBinLabel(7, "56, 220, C1");
    xa->SetBinLabel(8, "56, 220, FR");
    ya->SetBinLabel(8, "56, 220, FR");
  }

  h_trackCorr_vert = ibooker.book2D("track correlation vert", "ctpps_common_rp_vert", 8, -0.5, 7.5, 8, -0.5, 7.5);
  {
    TH2F *hist = h_trackCorr_vert->getTH2F();
    TAxis *xa = hist->GetXaxis(), *ya = hist->GetYaxis();
    xa->SetBinLabel(1, "45, 210, FR-TP");
    ya->SetBinLabel(1, "45, 210, FR-TP");
    xa->SetBinLabel(2, "45, 210, FR-BT");
    ya->SetBinLabel(2, "45, 210, FR-BT");
    xa->SetBinLabel(3, "45, 220, FR-TP");
    ya->SetBinLabel(3, "45, 220, FR-TP");
    xa->SetBinLabel(4, "45, 220, FR-BT");
    ya->SetBinLabel(4, "45, 220, FR-BT");
    xa->SetBinLabel(5, "56, 210, FR-TP");
    ya->SetBinLabel(5, "56, 210, FR-TP");
    xa->SetBinLabel(6, "56, 210, FR-BT");
    ya->SetBinLabel(6, "56, 210, FR-BT");
    xa->SetBinLabel(7, "56, 220, FR-TP");
    ya->SetBinLabel(7, "56, 220, FR-TP");
    xa->SetBinLabel(8, "56, 220, FR-BT");
    ya->SetBinLabel(8, "56, 220, FR-BT");
  }
}

//----------------------------------------------------------------------------------------------------

CTPPSCommonDQMSource::ArmPlots::ArmPlots(DQMStore::IBooker &ibooker, int _id, bool makeProtonRecoPlots) : id(_id) {
  string name;
  CTPPSDetId(CTPPSDetId::sdTrackingStrip, id, 0).armName(name, CTPPSDetId::nShort);

  ibooker.setCurrentFolder("CTPPS/common/sector " + name);

  string title = "ctpps_common_sector_" + name;

  h_numRPWithTrack_top =
      ibooker.book1D("number of top RPs with tracks", title + ";number of top RPs with tracks", 5, -0.5, 4.5);
  h_numRPWithTrack_hor =
      ibooker.book1D("number of hor RPs with tracks", title + ";number of hor RPs with tracks", 5, -0.5, 4.5);
  h_numRPWithTrack_bot =
      ibooker.book1D("number of bot RPs with tracks", title + ";number of bot RPs with tracks", 5, -0.5, 4.5);

  h_trackCorr = ibooker.book2D("track correlation", title, 8, -0.5, 7.5, 8, -0.5, 7.5);
  TH2F *h_trackCorr_h = h_trackCorr->getTH2F();
  TAxis *xa = h_trackCorr_h->GetXaxis(), *ya = h_trackCorr_h->GetYaxis();
  xa->SetBinLabel(1, "210, FR-HR");
  ya->SetBinLabel(1, "210, FR-HR");
  xa->SetBinLabel(2, "210, FR-TP");
  ya->SetBinLabel(2, "210, FR-TP");
  xa->SetBinLabel(3, "210, FR-BT");
  ya->SetBinLabel(3, "210, FR-BT");
  xa->SetBinLabel(4, "220, NR-HR");
  ya->SetBinLabel(4, "220, NR-HR");
  xa->SetBinLabel(5, "220, C1");
  ya->SetBinLabel(5, "220, C1");
  xa->SetBinLabel(6, "220, FR-HR");
  ya->SetBinLabel(6, "220, FR-HR");
  xa->SetBinLabel(7, "220, FR-TP");
  ya->SetBinLabel(7, "220, FR-TP");
  xa->SetBinLabel(8, "220, FR-BT");
  ya->SetBinLabel(8, "220, FR-BT");

  h_trackCorr_overlap = ibooker.book2D("track correlation hor-vert overlaps", title, 8, -0.5, 7.5, 8, -0.5, 7.5);
  h_trackCorr_h = h_trackCorr_overlap->getTH2F();
  xa = h_trackCorr_h->GetXaxis();
  ya = h_trackCorr_h->GetYaxis();
  xa->SetBinLabel(1, "210, FR-HR");
  ya->SetBinLabel(1, "210, FR-HR");
  xa->SetBinLabel(2, "210, FR-TP");
  ya->SetBinLabel(2, "210, FR-TP");
  xa->SetBinLabel(3, "210, FR-BT");
  ya->SetBinLabel(3, "210, FR-BT");
  xa->SetBinLabel(4, "220, NR-HR");
  ya->SetBinLabel(4, "220, NR-HR");
  xa->SetBinLabel(5, "220, C1");
  ya->SetBinLabel(5, "220, C1");
  xa->SetBinLabel(6, "220, FR-HR");
  ya->SetBinLabel(6, "220, FR-HR");
  xa->SetBinLabel(7, "220, FR-TP");
  ya->SetBinLabel(7, "220, FR-TP");
  xa->SetBinLabel(8, "220, FR-BT");
  ya->SetBinLabel(8, "220, FR-BT");

  if (makeProtonRecoPlots) {
    h_proton_xi = ibooker.book1D("proton xi", title + ";xi", 100, 0., 0.3);
    h_proton_th_x = ibooker.book1D("proton theta st x", ";#theta^{*}_{x}   (rad)", 250, -500E-6, +500E-6);
    h_proton_th_y = ibooker.book1D("proton theta st y", ";#theta^{*}_{y}   (rad)", 250, -500E-6, +500E-6);
    h_proton_t = ibooker.book1D("proton t", title + ";|t|   GeV^{2}", 100, 0., 5.);
    h_proton_time = ibooker.book1D("proton time", title + ";time   (ns)", 100, -1., 1.);
  }

  for (const unsigned int rpDecId : {3, 22, 16, 23}) {
    unsigned int st = rpDecId / 10, rp = rpDecId % 10, rpFullDecId = id * 100 + rpDecId;
    CTPPSDetId rpId(CTPPSDetId::sdTrackingStrip, id, st, rp);
    string stName, rpName;
    rpId.stationName(stName, CTPPSDetId::nShort);
    rpId.rpName(rpName, CTPPSDetId::nShort);
    rpName = stName + "_" + rpName;

    const bool timingRP = (rpDecId == 22 || rpDecId == 16);

    if (timingRP) {
      timingRPPlots[rpFullDecId] = {
          ibooker.book1D(rpName + " - track x histogram", title + "/" + rpName + ";track x   (mm)", 200, 0., 40.),
          ibooker.book1D(
              rpName + " - track time histogram", title + "/" + rpName + ";track time   (ns)", 100, -25., +50.)};
    } else {
      trackingRPPlots[rpFullDecId] = {
          ibooker.book1D(rpName + " - track x histogram", title + "/" + rpName + ";track x   (mm)", 200, 0., 40.),
          ibooker.book1D(rpName + " - track y histogram", title + "/" + rpName + ";track y   (mm)", 200, -20., +20.)};
    }
  }
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSCommonDQMSource::CTPPSCommonDQMSource(const edm::ParameterSet &ps)
    : verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
      ctppsRecordToken(consumes<CTPPSRecord>(ps.getUntrackedParameter<edm::InputTag>("ctppsmetadata"))),
      tokenLocalTrackLite(
          consumes<vector<CTPPSLocalTrackLite>>(ps.getUntrackedParameter<edm::InputTag>("tagLocalTrackLite"))),
      tokenRecoProtons(
          consumes<std::vector<reco::ForwardProton>>(ps.getUntrackedParameter<InputTag>("tagRecoProtons"))),
      makeProtonRecoPlots_(ps.getParameter<bool>("makeProtonRecoPlots")),
      perLSsaving_(ps.getUntrackedParameter<bool>("perLSsaving", false)) {
  currentLS = 0;
  endLS = 0;
  rpstate.clear();
}

//----------------------------------------------------------------------------------------------------

CTPPSCommonDQMSource::~CTPPSCommonDQMSource() {}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {
  // global plots
  globalPlots.Init(ibooker);

  // loop over arms
  for (unsigned int arm = 0; arm < 2; arm++) {
    armPlots[arm] = ArmPlots(ibooker, arm, makeProtonRecoPlots_);
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::analyze(edm::Event const &event, edm::EventSetup const &eventSetup) {
  analyzeCTPPSRecord(event, eventSetup);
  analyzeTracks(event, eventSetup);

  if (makeProtonRecoPlots_)
    analyzeProtons(event, eventSetup);
}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::analyzeCTPPSRecord(edm::Event const &event, edm::EventSetup const &eventSetup) {
  Handle<CTPPSRecord> hCTPPSRecord;
  event.getByToken(ctppsRecordToken, hCTPPSRecord);

  if (!hCTPPSRecord.isValid()) {
    if (verbosity)
      LogProblem("CTPPSCommonDQMSource") << "ERROR in CTPPSCommonDQMSource::analyzeCTPPSRecord > input not available.";

    return;
  }

  auto &rpstate = *luminosityBlockCache(event.getLuminosityBlock().index());
  if (rpstate.empty()) {
    rpstate.reserve(CTPPSRecord::RomanPot::Last);
    for (uint8_t i = 0; i < CTPPSRecord::RomanPot::Last; ++i)
      rpstate.push_back(hCTPPSRecord->status(i));
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::analyzeTracks(edm::Event const &event, edm::EventSetup const &eventSetup) {
  // get event data
  Handle<vector<CTPPSLocalTrackLite>> hTracks;
  event.getByToken(tokenLocalTrackLite, hTracks);

  // check validity
  if (!hTracks.isValid()) {
    if (verbosity)
      LogProblem("CTPPSCommonDQMSource") << "ERROR in CTPPSCommonDQMSource::analyzeTracks > input not available.";

    return;
  }

  //------------------------------
  // collect indeces of RP with tracks, for each correlation plot
  set<signed int> s_rp_idx_global_hor, s_rp_idx_global_vert;
  map<unsigned int, set<signed int>> ms_rp_idx_arm;

  for (auto &tr : *hTracks) {
    const CTPPSDetId rpId(tr.rpId());
    const unsigned int arm = rpId.arm();
    const unsigned int stNum = rpId.station();
    const unsigned int rpNum = rpId.rp();
    const unsigned int stRPNum = stNum * 10 + rpNum;

    {
      signed int idx = -1;
      if (stRPNum == 3)
        idx = 0;
      if (stRPNum == 22)
        idx = 1;
      if (stRPNum == 16)
        idx = 2;
      if (stRPNum == 23)
        idx = 3;

      if (idx >= 0)
        s_rp_idx_global_hor.insert(4 * arm + idx);
    }

    {
      signed int idx = -1;
      if (stRPNum == 4)
        idx = 0;
      if (stRPNum == 5)
        idx = 1;
      if (stRPNum == 24)
        idx = 2;
      if (stRPNum == 25)
        idx = 3;

      if (idx >= 0)
        s_rp_idx_global_vert.insert(4 * arm + idx);
    }

    {
      signed int idx = -1;
      if (stRPNum == 3)
        idx = 0;
      if (stRPNum == 4)
        idx = 1;
      if (stRPNum == 5)
        idx = 2;
      if (stRPNum == 22)
        idx = 3;
      if (stRPNum == 16)
        idx = 4;
      if (stRPNum == 23)
        idx = 5;
      if (stRPNum == 24)
        idx = 6;
      if (stRPNum == 25)
        idx = 7;

      const signed int hor = ((rpNum == 2) || (rpNum == 3) || (rpNum == 6)) ? 1 : 0;

      if (idx >= 0)
        ms_rp_idx_arm[arm].insert(idx * 10 + hor);
    }
  }

  //------------------------------
  // Global Plots

  globalPlots.events_per_bx->Fill(event.bunchCrossing());
  globalPlots.events_per_bx_short->Fill(event.bunchCrossing());

  for (const auto &idx1 : s_rp_idx_global_hor)
    for (const auto &idx2 : s_rp_idx_global_hor)
      globalPlots.h_trackCorr_hor->Fill(idx1, idx2);

  for (const auto &idx1 : s_rp_idx_global_vert)
    for (const auto &idx2 : s_rp_idx_global_vert)
      globalPlots.h_trackCorr_vert->Fill(idx1, idx2);

  //------------------------------
  // Arm Plots

  map<unsigned int, set<unsigned int>> mTop, mHor, mBot;

  for (auto &tr : *hTracks) {
    CTPPSDetId rpId(tr.rpId());
    const unsigned int rpNum = rpId.rp();
    const unsigned int armIdx = rpId.arm();

    if (rpNum == 0 || rpNum == 4)
      mTop[armIdx].insert(rpId);
    if (rpNum == 2 || rpNum == 3 || rpNum == 6)
      mHor[armIdx].insert(rpId);
    if (rpNum == 1 || rpNum == 5)
      mBot[armIdx].insert(rpId);

    auto &ap = armPlots[rpId.arm()];
    unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();

    // fill in reference tracking-RP plots
    {
      auto it = ap.trackingRPPlots.find(rpDecId);
      if (it != ap.trackingRPPlots.end()) {
        it->second.h_x->Fill(tr.x());
        it->second.h_y->Fill(tr.y());
      }
    }

    // fill in reference timing-RP plots
    {
      auto it = ap.timingRPPlots.find(rpDecId);
      if (it != ap.timingRPPlots.end()) {
        it->second.h_x->Fill(tr.x());
        it->second.h_time->Fill(tr.time());
      }
    }
  }

  for (auto &p : armPlots) {
    p.second.h_numRPWithTrack_top->Fill(mTop[p.first].size());
    p.second.h_numRPWithTrack_hor->Fill(mHor[p.first].size());
    p.second.h_numRPWithTrack_bot->Fill(mBot[p.first].size());
  }

  //------------------------------
  // Correlation plots

  for (const auto &ap : ms_rp_idx_arm) {
    auto &plots = armPlots[ap.first];

    for (const auto &idx1 : ap.second) {
      for (const auto &idx2 : ap.second) {
        plots.h_trackCorr->Fill(idx1 / 10, idx2 / 10);

        if ((idx1 % 10) != (idx2 % 10))
          plots.h_trackCorr_overlap->Fill(idx1 / 10, idx2 / 10);
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::analyzeProtons(edm::Event const &event, edm::EventSetup const &eventSetup) {
  // get event data
  Handle<vector<reco::ForwardProton>> hRecoProtons;
  event.getByToken(tokenRecoProtons, hRecoProtons);

  // check validity
  if (!hRecoProtons.isValid()) {
    if (verbosity)
      LogProblem("CTPPSCommonDQMSource") << "ERROR in CTPPSCommonDQMSource::analyzeProtons > input not available.";

    return;
  }

  // loop over protons
  for (auto &p : *hRecoProtons) {
    if (!p.validFit())
      continue;

    signed int armIndex = -1;
    if (p.lhcSector() == reco::ForwardProton::LHCSector::sector45)
      armIndex = 0;
    if (p.lhcSector() == reco::ForwardProton::LHCSector::sector56)
      armIndex = 1;
    if (armIndex < 0)
      continue;

    auto &plots = armPlots[armIndex];

    plots.h_proton_xi->Fill(p.xi());
    plots.h_proton_th_x->Fill(p.thetaX());
    plots.h_proton_th_y->Fill(p.thetaY());
    plots.h_proton_t->Fill(fabs(p.t()));
    plots.h_proton_time->Fill(p.time());
  }
}

//----------------------------------------------------------------------------------------------------

std::shared_ptr<std::vector<int>> CTPPSCommonDQMSource::globalBeginLuminosityBlock(const edm::LuminosityBlock &,
                                                                                   const edm::EventSetup &) const {
  return std::make_shared<std::vector<int>>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::globalEndLuminosityBlock(const edm::LuminosityBlock &iLumi, const edm::EventSetup &c) {
  auto const &rpstate = *luminosityBlockCache(iLumi.index());
  auto currentLS = iLumi.id().luminosityBlock();
  if (!perLSsaving_) {
    for (std::vector<int>::size_type i = 0; i < rpstate.size(); i++)
      globalPlots.RPState->setBinContent(currentLS, i + 1, rpstate[i]);
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSCommonDQMSource);
