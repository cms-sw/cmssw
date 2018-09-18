/****************************************************************************
 *
 * This is a part of TotemDQM and TOTEM offline software.
 * Authors:
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/OnlineMetaData/interface/CTPPSRecord.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class CTPPSCommonDQMSource: public DQMEDAnalyzer
{
  public:

    CTPPSCommonDQMSource(const edm::ParameterSet& ps);

    ~CTPPSCommonDQMSource() override;

  protected:

    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
    void beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& c) override;
    void endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& c) override;

  private:
    unsigned int verbosity;
    const static int MAX_LUMIS = 6000;
    const static int MAX_VBINS = 18;

    edm::EDGetTokenT< std::vector<CTPPSLocalTrackLite> > tokenLocalTrackLite;
    edm::EDGetTokenT<CTPPSRecord> ctppsRecordToken;

    int currentLS;
    int endLS;

    std::vector<int> rpstate;

    /// plots related to the whole system
    struct GlobalPlots
    {
      MonitorElement *RPState = nullptr;
      MonitorElement *events_per_bx = nullptr, *events_per_bx_short = nullptr;
      MonitorElement *h_trackCorr_hor = nullptr, *h_trackCorr_vert = nullptr;

      void Init(DQMStore::IBooker &ibooker);
    };

    GlobalPlots globalPlots;

    /// plots related to one arm
    struct ArmPlots
    {
      int id;

      MonitorElement *h_numRPWithTrack_top=nullptr, *h_numRPWithTrack_hor=nullptr, *h_numRPWithTrack_bot=nullptr;
      MonitorElement *h_trackCorr=nullptr, *h_trackCorr_overlap=nullptr;

      ArmPlots(){}

      ArmPlots(DQMStore::IBooker &ibooker, int _id);
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

void CTPPSCommonDQMSource::GlobalPlots::Init(DQMStore::IBooker &ibooker)
{
  ibooker.setCurrentFolder("CTPPS/common");

  events_per_bx = ibooker.book1D("events per BX", "rp;Event.BX", 4002, -1.5, 4000. + 0.5);
  events_per_bx_short = ibooker.book1D("events per BX (short)", "rp;Event.BX", 102, -1.5, 100. + 0.5);

  RPState = ibooker.book2D("rpstate per LS","RP State per Lumisection;Luminosity Section;",MAX_LUMIS, 0, MAX_LUMIS, MAX_VBINS, 0., MAX_VBINS);
  {
    TH2F* hist = RPState->getTH2F();
    hist->SetCanExtend(TH1::kAllAxes);
    TAxis* ya = hist->GetYaxis();
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

  h_trackCorr_hor = ibooker.book2D("track correlation hor", "ctpps_common_rp_hor", 6, -0.5, 5.5, 6, -0.5, 5.5);
  {
    TH2F* hist = h_trackCorr_hor->getTH2F();
    TAxis* xa = hist->GetXaxis(), *ya = hist->GetYaxis();
    xa->SetBinLabel(1, "45, 210, far"); ya->SetBinLabel(1, "45, 210, far");
    xa->SetBinLabel(2, "45, 220, far"); ya->SetBinLabel(2, "45, 220, far");
    xa->SetBinLabel(3, "45, 220, cyl"); ya->SetBinLabel(3, "45, 220, cyl");
    xa->SetBinLabel(4, "56, 210, far"); ya->SetBinLabel(4, "56, 210, far");
    xa->SetBinLabel(5, "56, 220, far"); ya->SetBinLabel(5, "56, 220, far");
    xa->SetBinLabel(6, "56, 220, cyl"); ya->SetBinLabel(6, "56, 220, cyl");
  }

  h_trackCorr_vert = ibooker.book2D("track correlation vert", "ctpps_common_rp_vert", 8, -0.5, 7.5, 8, -0.5, 7.5);
  {
    TH2F* hist = h_trackCorr_vert->getTH2F();
    TAxis* xa = hist->GetXaxis(), *ya = hist->GetYaxis();
    xa->SetBinLabel(1, "45, 210, far, top"); ya->SetBinLabel(1, "45, 210, far, top");
    xa->SetBinLabel(2, "45, 210, far, bot"); ya->SetBinLabel(2, "45, 210, far, bot");
    xa->SetBinLabel(3, "45, 220, far, top"); ya->SetBinLabel(3, "45, 220, far, top");
    xa->SetBinLabel(4, "45, 220, far, bot"); ya->SetBinLabel(4, "45, 220, far, bot");
    xa->SetBinLabel(5, "56, 210, far, top"); ya->SetBinLabel(5, "56, 210, far, top");
    xa->SetBinLabel(6, "56, 210, far, bot"); ya->SetBinLabel(6, "56, 210, far, bot");
    xa->SetBinLabel(7, "56, 220, far, top"); ya->SetBinLabel(7, "56, 220, far, top");
    xa->SetBinLabel(8, "56, 220, far, bot"); ya->SetBinLabel(8, "56, 220, far, bot");
  }
}

//----------------------------------------------------------------------------------------------------

CTPPSCommonDQMSource::ArmPlots::ArmPlots(DQMStore::IBooker &ibooker, int _id) : id(_id)
{
  string name;
  CTPPSDetId(CTPPSDetId::sdTrackingStrip, id, 0).armName(name, CTPPSDetId::nShort);

  ibooker.setCurrentFolder("CTPPS/common/sector " + name);

  string title = "ctpps_common_sector_" + name;

  h_numRPWithTrack_top = ibooker.book1D("number of top RPs with tracks", title+";number of top RPs with tracks", 5, -0.5, 4.5);
  h_numRPWithTrack_hor = ibooker.book1D("number of hor RPs with tracks", title+";number of hor RPs with tracks", 5, -0.5, 4.5);
  h_numRPWithTrack_bot = ibooker.book1D("number of bot RPs with tracks", title+";number of bot RPs with tracks", 5, -0.5, 4.5);

  h_trackCorr = ibooker.book2D("track correlation", title, 7, -0.5, 6.5, 7, -0.5, 6.5);
  TH2F *h_trackCorr_h = h_trackCorr->getTH2F();
  TAxis *xa = h_trackCorr_h->GetXaxis(), *ya = h_trackCorr_h->GetYaxis();
  xa->SetBinLabel( 1, "210, far, hor"); ya->SetBinLabel( 1, "210, far, hor");
  xa->SetBinLabel( 2, "210, far, top"); ya->SetBinLabel( 2, "210, far, top");
  xa->SetBinLabel( 3, "210, far, bot"); ya->SetBinLabel( 3, "210, far, bot");
  xa->SetBinLabel( 4, "220, cyl"     ); ya->SetBinLabel( 4, "220, cyl"     );
  xa->SetBinLabel( 5, "220, far, hor"); ya->SetBinLabel( 5, "220, far, hor");
  xa->SetBinLabel( 6, "220, far, top"); ya->SetBinLabel( 6, "220, far, top");
  xa->SetBinLabel( 7, "220, far, bot"); ya->SetBinLabel( 7, "220, far, bot");

  h_trackCorr_overlap = ibooker.book2D("track correlation hor-vert overlaps", title, 7, -0.5, 6.5, 7, -0.5, 6.5);
  h_trackCorr_h = h_trackCorr_overlap->getTH2F();
  xa = h_trackCorr_h->GetXaxis(); ya = h_trackCorr_h->GetYaxis();
  xa->SetBinLabel( 1, "210, far, hor"); ya->SetBinLabel( 1, "210, far, hor");
  xa->SetBinLabel( 2, "210, far, top"); ya->SetBinLabel( 2, "210, far, top");
  xa->SetBinLabel( 3, "210, far, bot"); ya->SetBinLabel( 3, "210, far, bot");
  xa->SetBinLabel( 4, "220, cyl"     ); ya->SetBinLabel( 4, "220, cyl"     );
  xa->SetBinLabel( 5, "220, far, hor"); ya->SetBinLabel( 5, "220, far, hor");
  xa->SetBinLabel( 6, "220, far, top"); ya->SetBinLabel( 6, "220, far, top");
  xa->SetBinLabel( 7, "220, far, bot"); ya->SetBinLabel( 7, "220, far, bot");
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSCommonDQMSource::CTPPSCommonDQMSource(const edm::ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0))
{
  tokenLocalTrackLite = consumes< vector<CTPPSLocalTrackLite> >(ps.getParameter<edm::InputTag>("tagLocalTrackLite"));
  ctppsRecordToken = consumes<CTPPSRecord>(ps.getUntrackedParameter<edm::InputTag>("ctppsmetadata"));

  currentLS = 0;
  endLS = 0;
  rpstate.clear();

}

//----------------------------------------------------------------------------------------------------

CTPPSCommonDQMSource::~CTPPSCommonDQMSource()
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &)
{
  // global plots
  globalPlots.Init(ibooker);

  // loop over arms
  for (unsigned int arm = 0; arm < 2; arm++)
  {
    armPlots[arm] = ArmPlots(ibooker, arm);
  }
}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{

  // get CTPPS Event Record
  Handle<CTPPSRecord> rp;
  event.getByToken(ctppsRecordToken, rp);

  /*
     RP State (HV & LV & Insertion):
     0 -> not used
     1 -> bad
     2 -> warning
     3 -> ok

     CTPPSRecord Order:
     RP name: RP_45_210_FR_BT
     RP name: RP_45_210_FR_HR
     RP name: RP_45_210_FR_TP
     RP name: RP_45_220_C1
     RP name: RP_45_220_FR_BT
     RP name: RP_45_220_FR_HR
     RP name: RP_45_220_FR_TP
     RP name: RP_45_220_NR_BT
     RP name: RP_45_220_NR_TP
     RP name: RP_56_210_FR_BT
     RP name: RP_56_210_FR_HR
     RP name: RP_56_210_FR_TP
     RP name: RP_56_220_C1
     RP name: RP_56_220_FR_BT
     RP name: RP_56_220_FR_HR
     RP name: RP_56_220_FR_TP
     RP name: RP_56_220_NR_BT
     RP name: RP_56_220_NR_TP
     */

  if(currentLS > endLS){
    rpstate.clear();
    for (uint8_t i = 0; i < CTPPSRecord::RomanPot::Last; ++i) {
      rpstate.push_back(rp->status(i));
    }
    endLS=currentLS;
  }  

  // get event data
  Handle< vector<CTPPSLocalTrackLite> > tracks;
  event.getByToken(tokenLocalTrackLite, tracks);

  // check validity
  bool valid = true;
  valid &= tracks.isValid();

  if (!valid)
  {
    if (verbosity)
    {
      LogProblem("CTPPSCommonDQMSource")
	<< "    trackLites.isValid = " << tracks.isValid();
    }

    return;
  }

  //------------------------------
  // collect indeces of RP with tracks, for each correlation plot
  set<signed int> s_rp_idx_global_hor, s_rp_idx_global_vert;
  map<unsigned int, set<signed int>> ms_rp_idx_arm;

  for (auto &tr : *tracks)
  {
    const CTPPSDetId rpId(tr.getRPId());
    const unsigned int arm = rpId.arm();
    const unsigned int stNum = rpId.station();
    const unsigned int rpNum = rpId.rp();
    const unsigned int stRPNum = stNum * 10 + rpNum;

    {
      signed int idx = -1;
      if (stRPNum ==  3) idx = 0;
      if (stRPNum == 23) idx = 1;
      if (stRPNum == 16) idx = 2;

      if (idx >= 0)
	s_rp_idx_global_hor.insert(3*arm + idx);
    }

    {
      signed int idx = -1;
      if (stRPNum ==  4) idx = 0;
      if (stRPNum ==  5) idx = 1;
      if (stRPNum == 24) idx = 2;
      if (stRPNum == 25) idx = 3;

      if (idx >= 0)
	s_rp_idx_global_vert.insert(4*arm + idx);
    }

    {
      signed int idx = -1;
      if (stRPNum ==  3) idx = 0;
      if (stRPNum ==  4) idx = 1;
      if (stRPNum ==  5) idx = 2;
      if (stRPNum == 16) idx = 3;
      if (stRPNum == 23) idx = 4;
      if (stRPNum == 24) idx = 5;
      if (stRPNum == 25) idx = 6;

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

  for (auto &tr : *tracks)
  {
    CTPPSDetId rpId(tr.getRPId());
    const unsigned int rpNum = rpId.rp();
    const unsigned int armIdx = rpId.arm();

    if (rpNum == 0 || rpNum == 4)
      mTop[armIdx].insert(rpId);
    if (rpNum == 2 || rpNum == 3 || rpNum == 6)
      mHor[armIdx].insert(rpId);
    if (rpNum == 1 || rpNum == 5)
      mBot[armIdx].insert(rpId);
  }

  for (auto &p : armPlots)
  {
    p.second.h_numRPWithTrack_top->Fill(mTop[p.first].size());
    p.second.h_numRPWithTrack_hor->Fill(mHor[p.first].size());
    p.second.h_numRPWithTrack_bot->Fill(mBot[p.first].size());
  }

  // track RP correlation
  for (const auto &ap : ms_rp_idx_arm)
  {
    auto &plots = armPlots[ap.first];

    for (const auto &idx1 : ap.second)
    {
      for (const auto &idx2 : ap.second)
      {
	plots.h_trackCorr->Fill(idx1/10, idx2/10);

	if ((idx1 % 10) != (idx2 % 10))
	  plots.h_trackCorr_overlap->Fill(idx1/10, idx2/10);
      }
    }
  }

}
//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& c) {

  currentLS = iLumi.id().luminosityBlock();

}

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& c) {

  for(std::vector<int>::size_type i=0; i<rpstate.size();i++){
    globalPlots.RPState->setBinContent(currentLS, i+1, rpstate[i]);
  }

  endLS = iLumi.luminosityBlock();
  rpstate.clear();

}
//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSCommonDQMSource);
