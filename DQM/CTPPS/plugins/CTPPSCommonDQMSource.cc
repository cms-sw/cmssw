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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

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

  private:
    unsigned int verbosity;

    edm::EDGetTokenT< std::vector<CTPPSLocalTrackLite> > tokenLocalTrackLite;

    /// plots related to the whole system
    struct GlobalPlots
    {
      MonitorElement *events_per_bx = nullptr, *events_per_bx_short = nullptr;
      MonitorElement *h_trackCorr_hor = nullptr;

      // TODO: add h_trackCorr_vert

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

//----------------------------------------------------------------------------------------------------

void CTPPSCommonDQMSource::GlobalPlots::Init(DQMStore::IBooker &ibooker)
{
  ibooker.setCurrentFolder("CTPPS/common");

  events_per_bx = ibooker.book1D("events per BX", "rp;Event.BX", 4002, -1.5, 4000. + 0.5);
  events_per_bx_short = ibooker.book1D("events per BX (short)", "rp;Event.BX", 102, -1.5, 100. + 0.5);

  h_trackCorr_hor = ibooker.book2D("track correlation RP-210-hor", "rp, 210, hor", 4, -0.5, 3.5, 4, -0.5, 3.5);
  TH2F *hist = h_trackCorr_hor->getTH2F();
  TAxis *xa = hist->GetXaxis(), *ya = hist->GetYaxis();
  xa->SetBinLabel(1, "45, 210, near"); ya->SetBinLabel(1, "45, 210, near");
  xa->SetBinLabel(2, "45, 210, far"); ya->SetBinLabel(2, "45, 210, far");
  xa->SetBinLabel(3, "56, 210, near"); ya->SetBinLabel(3, "56, 210, near");
  xa->SetBinLabel(4, "56, 210, far"); ya->SetBinLabel(4, "56, 210, far");
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

  // TODO: add timing RPs
  h_trackCorr = ibooker.book2D("track RP correlation", title, 13, -0.5, 12.5, 13, -0.5, 12.5);
  TH2F *h_trackCorr_h = h_trackCorr->getTH2F();
  TAxis *xa = h_trackCorr_h->GetXaxis(), *ya = h_trackCorr_h->GetYaxis();
  xa->SetBinLabel(1, "210, near, top"); ya->SetBinLabel(1, "210, near, top");
  xa->SetBinLabel(2, "bot"); ya->SetBinLabel(2, "bot");
  xa->SetBinLabel(3, "hor"); ya->SetBinLabel(3, "hor");
  xa->SetBinLabel(4, "far, hor"); ya->SetBinLabel(4, "far, hor");
  xa->SetBinLabel(5, "top"); ya->SetBinLabel(5, "top");
  xa->SetBinLabel(6, "bot"); ya->SetBinLabel(6, "bot");
  xa->SetBinLabel(8, "220, near, top"); ya->SetBinLabel(8, "220, near, top");
  xa->SetBinLabel(9, "bot"); ya->SetBinLabel(9, "bot");
  xa->SetBinLabel(10, "hor"); ya->SetBinLabel(10, "hor");
  xa->SetBinLabel(11, "far, hor"); ya->SetBinLabel(11, "far, hor");
  xa->SetBinLabel(12, "top"); ya->SetBinLabel(12, "top");
  xa->SetBinLabel(13, "bot"); ya->SetBinLabel(13, "bot");

  // TODO: add timing RPs
  h_trackCorr_overlap = ibooker.book2D("track RP correlation hor-vert overlaps", title, 13, -0.5, 12.5, 13, -0.5, 12.5);
  h_trackCorr_h = h_trackCorr_overlap->getTH2F();
  xa = h_trackCorr_h->GetXaxis(); ya = h_trackCorr_h->GetYaxis();
  xa->SetBinLabel(1, "210, near, top"); ya->SetBinLabel(1, "210, near, top");
  xa->SetBinLabel(2, "bot"); ya->SetBinLabel(2, "bot");
  xa->SetBinLabel(3, "hor"); ya->SetBinLabel(3, "hor");
  xa->SetBinLabel(4, "far, hor"); ya->SetBinLabel(4, "far, hor");
  xa->SetBinLabel(5, "top"); ya->SetBinLabel(5, "top");
  xa->SetBinLabel(6, "bot"); ya->SetBinLabel(6, "bot");
  xa->SetBinLabel(8, "220, near, top"); ya->SetBinLabel(8, "220, near, top");
  xa->SetBinLabel(9, "bot"); ya->SetBinLabel(9, "bot");
  xa->SetBinLabel(10, "hor"); ya->SetBinLabel(10, "hor");
  xa->SetBinLabel(11, "far, hor"); ya->SetBinLabel(11, "far, hor");
  xa->SetBinLabel(12, "top"); ya->SetBinLabel(12, "top");
  xa->SetBinLabel(13, "bot"); ya->SetBinLabel(13, "bot");
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSCommonDQMSource::CTPPSCommonDQMSource(const edm::ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0))
{
  tokenLocalTrackLite = consumes< vector<CTPPSLocalTrackLite> >(ps.getParameter<edm::InputTag>("tagLocalTrackLite"));
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
  // Global Plots

  globalPlots.events_per_bx->Fill(event.bunchCrossing());
  globalPlots.events_per_bx_short->Fill(event.bunchCrossing());

  for (auto &tr1 : *tracks)
  {
    CTPPSDetId rpId1(tr1.getRPId());
    unsigned int arm1 = rpId1.arm();
    unsigned int stNum1 = rpId1.station();
    unsigned int rpNum1 = rpId1.rp();
    if (stNum1 != 0 || (rpNum1 != 2 && rpNum1 != 3))
      continue;
    unsigned int idx1 = arm1*2 + rpNum1-2;

    for (auto &tr2 : *tracks)
    {
      CTPPSDetId rpId2(tr2.getRPId());
      unsigned int arm2 = rpId2.arm();
      unsigned int stNum2 = rpId2.station();
      unsigned int rpNum2 = rpId2.rp();
      if (stNum2 != 0 || (rpNum2 != 2 && rpNum2 != 3))
        continue;
      unsigned int idx2 = arm2*2 + rpNum2-2;
  
      globalPlots.h_trackCorr_hor->Fill(idx1, idx2); 
    }
  }

  //------------------------------
  // Arm Plots
  {
    map<unsigned int, set<unsigned int>> mTop, mHor, mBot;

    for (auto &tr : *tracks)
    {
      CTPPSDetId rpId(tr.getRPId());
      const unsigned int rpNum = rpId.rp();
      const unsigned int armIdx = rpId.arm();

      if (rpNum == 0 || rpNum == 4)
        mTop[armIdx].insert(rpId);
      if (rpNum == 2 || rpNum == 3)
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
    for (auto &tr1 : *tracks)
    {
      // TODO: check whether this rule works for timing RPs
      // TODO: encapsulate formulae in a function ?
      CTPPSDetId rpId1(tr1.getRPId());
      unsigned int arm1 = rpId1.arm();
      unsigned int stNum1 = rpId1.station();
      unsigned int rpNum1 = rpId1.rp();
      unsigned int idx1 = stNum1/2 * 7 + rpNum1;
      bool hor1 = (rpNum1 == 2 || rpNum1 == 3);

      ArmPlots &ap = armPlots[arm1];

      for (auto &tr2 : *tracks)
      {
        CTPPSDetId rpId2(tr2.getRPId());
        unsigned int arm2 = rpId2.arm();
        unsigned int stNum2 = rpId2.station();
        unsigned int rpNum2 = rpId2.rp();
        unsigned int idx2 = stNum2/2 * 7 + rpNum2;
        bool hor2 = (rpNum2 == 2 || rpNum2 == 3);

        if (arm1 != arm2)
          continue;

        ap.h_trackCorr->Fill(idx1, idx2); 
        
        if (hor1 != hor2)
          ap.h_trackCorr_overlap->Fill(idx1, idx2); 
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSCommonDQMSource);
