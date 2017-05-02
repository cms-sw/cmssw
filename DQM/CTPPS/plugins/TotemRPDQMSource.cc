/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Rafał Leszko (rafal.leszko@gmail.com)
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

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSReco/interface/TotemRPCluster.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"
//#include "RecoTotemRP/RPRecoDataFormats/interface/RPMulFittedTrackCollection.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"

#include <string>

//----------------------------------------------------------------------------------------------------
 
class TotemRPDQMSource: public DQMEDAnalyzer
{
  public:
    TotemRPDQMSource(const edm::ParameterSet& ps);
    virtual ~TotemRPDQMSource();
  
  protected:
    void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
    void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override;
    void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup) override;
    void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  private:
    unsigned int verbosity;

    edm::EDGetTokenT< edm::DetSetVector<TotemVFATStatus> > tokenStatus;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPDigi> > tokenDigi;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPCluster> > tokenCluster;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPRecHit> > tokenRecHit;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPUVPattern> > tokenUVPattern;
    edm::EDGetTokenT< edm::DetSetVector<TotemRPLocalTrack> > tokenLocalTrack;
    //edm::EDGetTokenT< RPMulFittedTrackCollection > tokenMultiTrackColl;

    /// plots related to the whole system
    struct GlobalPlots
    {
      MonitorElement *events_per_bx = NULL, *events_per_bx_short = NULL;
      MonitorElement *h_trackCorr_hor = NULL;

      void Init(DQMStore::IBooker &ibooker);
    };

    GlobalPlots globalPlots;

    /// plots related to one (anti)diagonal
    struct DiagonalPlots
    {
      int id;

      MonitorElement *h_lrc_x_d=NULL, *h_lrc_x_n=NULL, *h_lrc_x_f=NULL;
      MonitorElement *h_lrc_y_d=NULL, *h_lrc_y_n=NULL, *h_lrc_y_f=NULL;

      DiagonalPlots() {}

      DiagonalPlots(DQMStore::IBooker &ibooker, int _id);
    };

    std::map<unsigned int, DiagonalPlots> diagonalPlots;

    /// plots related to one arm
    struct ArmPlots
    {
      int id;

      MonitorElement *h_numRPWithTrack_top=NULL, *h_numRPWithTrack_hor=NULL, *h_numRPWithTrack_bot=NULL;
      MonitorElement *h_trackCorr=NULL, *h_trackCorr_overlap=NULL;

      ArmPlots(){}

      ArmPlots(DQMStore::IBooker &ibooker, int _id);
    };

    std::map<unsigned int, ArmPlots> armPlots;

    /// plots related to one station
    struct StationPlots
    {
      StationPlots() {}
      StationPlots(DQMStore::IBooker &ibooker, int _id);
    };

    std::map<unsigned int, StationPlots> stationPlots;

    /// plots related to one RP
    struct PotPlots
    {
      MonitorElement *vfat_problem=NULL, *vfat_missing=NULL, *vfat_ec_bc_error=NULL, *vfat_corruption=NULL;

      MonitorElement *activity=NULL, *activity_u=NULL, *activity_v=NULL;
      MonitorElement *activity_per_bx=NULL, *activity_per_bx_short=NULL;
      MonitorElement *hit_plane_hist=NULL;
      MonitorElement *patterns_u=NULL, *patterns_v=NULL;
      MonitorElement *h_planes_fit_u=NULL, *h_planes_fit_v=NULL;
      MonitorElement *event_category=NULL;
      MonitorElement *trackHitsCumulativeHist=NULL;
      MonitorElement *track_u_profile=NULL, *track_v_profile=NULL;

      PotPlots() {}
      PotPlots(DQMStore::IBooker &ibooker, unsigned int id);
    };

    std::map<unsigned int, PotPlots> potPlots;

    /// plots related to one RP plane
    struct PlanePlots
    {
      MonitorElement *digi_profile_cumulative = NULL;
      MonitorElement *cluster_profile_cumulative = NULL;
      MonitorElement *hit_multiplicity = NULL;
      MonitorElement *cluster_size = NULL;
      MonitorElement *efficiency_num = NULL, *efficiency_den = NULL;

      PlanePlots() {}
      PlanePlots(DQMStore::IBooker &ibooker, unsigned int id);
    };

    std::map<unsigned int, PlanePlots> planePlots;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::GlobalPlots::Init(DQMStore::IBooker &ibooker)
{
  ibooker.setCurrentFolder("CTPPS/TrackingStrip");

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

TotemRPDQMSource::DiagonalPlots::DiagonalPlots(DQMStore::IBooker &ibooker, int _id) : id(_id)
{
  bool top45 = id & 2;
  bool top56 = id & 1;
  bool diag = (top45 != top56);

  char name[50];
  sprintf(name, "%s 45%s - 56%s",
    (diag) ? "diagonal" : "antidiagonal",
    (top45) ? "top" : "bot",
    (top56) ? "top" : "bot"
  );

  ibooker.setCurrentFolder(string("CTPPS/TrackingStrip/") + name);

  // TODO: define ranges! If defined automatically, can lead to problems when histograms are merged from several instances of the module.
  h_lrc_x_d = ibooker.book2D("dx left vs right", string(name) + " : dx left vs. right, histogram;#Delta x_{45};#Delta x_{56}", 50, 0., 0., 50, 0., 0.);
  h_lrc_x_n = ibooker.book2D("xn left vs right", string(name) + " : xn left vs. right, histogram;x^{N}_{45};x^{N}_{56}", 50, 0., 0., 50, 0., 0.);
  h_lrc_x_f = ibooker.book2D("xf left vs right", string(name) + " : xf left vs. right, histogram;x^{F}_{45};x^{F}_{56}", 50, 0., 0., 50, 0., 0.);

  h_lrc_y_d = ibooker.book2D("dy left vs right", string(name) + " : dy left vs. right, histogram;#Delta y_{45};#Delta y_{56}", 50, 0., 0., 50, 0., 0.);
  h_lrc_y_n = ibooker.book2D("yn left vs right", string(name) + " : yn left vs. right, histogram;y^{N}_{45};y^{N}_{56}", 50, 0., 0., 50, 0., 0.);
  h_lrc_y_f = ibooker.book2D("yf left vs right", string(name) + " : yf left vs. right, histogram;y^{F}_{45};y^{F}_{56}", 50, 0., 0., 50, 0., 0.);
}

//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::ArmPlots::ArmPlots(DQMStore::IBooker &ibooker, int _id) : id(_id)
{
  string path;
  TotemRPDetId(id).armName(path, TotemRPDetId::nPath);
  ibooker.setCurrentFolder(path);

  string title;
  TotemRPDetId(id).armName(title, TotemRPDetId::nFull);

  h_numRPWithTrack_top = ibooker.book1D("number of top RPs with tracks", title+";number of top RPs with tracks", 5, -0.5, 4.5);
  h_numRPWithTrack_hor = ibooker.book1D("number of hor RPs with tracks", title+";number of hor RPs with tracks", 5, -0.5, 4.5);
  h_numRPWithTrack_bot = ibooker.book1D("number of bot RPs with tracks", title+";number of bot RPs with tracks", 5, -0.5, 4.5);

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

TotemRPDQMSource::StationPlots::StationPlots(DQMStore::IBooker &ibooker, int id) 
{
  string path;
  TotemRPDetId(id).stationName(path, TotemRPDetId::nPath);
  ibooker.setCurrentFolder(path);
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::PotPlots::PotPlots(DQMStore::IBooker &ibooker, unsigned int id)
{
  string path;
  TotemRPDetId(id).rpName(path, TotemRPDetId::nPath);
  ibooker.setCurrentFolder(path);

  string title;
  TotemRPDetId(id).rpName(title, TotemRPDetId::nFull);

  vfat_problem = ibooker.book2D("vfats with any problem", title+";plane;vfat index", 10, -0.5, 9.5, 4, -0.5, 3.5);
  vfat_missing = ibooker.book2D("vfats missing", title+";plane;vfat index", 10, -0.5, 9.5, 4, -0.5, 3.5);
  vfat_ec_bc_error = ibooker.book2D("vfats with EC or BC error", title+";plane;vfat index", 10, -0.5, 9.5, 4, -0.5, 3.5);
  vfat_corruption = ibooker.book2D("vfats with data corruption", title+";plane;vfat index", 10, -0.5, 9.5, 4, -0.5, 3.5);

  activity = ibooker.book1D("active planes", title+";number of active planes", 11, -0.5, 10.5);
  activity_u = ibooker.book1D("active planes U", title+";number of active U planes", 11, -0.5, 10.5);
  activity_v = ibooker.book1D("active planes V", title+";number of active V planes", 11, -0.5, 10.5);

  activity_per_bx = ibooker.book1D("activity per BX", title+";Event.BX", 4002, -1.5, 4000. + 0.5);
  activity_per_bx_short = ibooker.book1D("activity per BX (short)", title+";Event.BX", 102, -1.5, 100. + 0.5);

  hit_plane_hist = ibooker.book2D("activity in planes (2D)", title+";plane number;strip number", 10, -0.5, 9.5, 32, -0.5, 511.5);

  patterns_u = ibooker.book1D("recognized patterns U", title+";number of recognized U patterns", 11, -0.5, 10.5); 
  patterns_v = ibooker.book1D("recognized patterns V", title+";number of recognized V patterns", 11, -0.5, 10.5); 

  h_planes_fit_u = ibooker.book1D("planes contributing to fit U", title+";number of planes contributing to U fit", 6, -0.5, 5.5);
  h_planes_fit_v = ibooker.book1D("planes contributing to fit V", title+";number of planes contributing to V fit", 6, -0.5, 5.5);

  event_category = ibooker.book1D("event category", title+";event category", 5, -0.5, 4.5);
  TH1F *event_category_h = event_category->getTH1F();
  event_category_h->GetXaxis()->SetBinLabel(1, "empty");
  event_category_h->GetXaxis()->SetBinLabel(2, "insufficient");
  event_category_h->GetXaxis()->SetBinLabel(3, "single-track");
  event_category_h->GetXaxis()->SetBinLabel(4, "multi-track");
  event_category_h->GetXaxis()->SetBinLabel(5, "shower");

  trackHitsCumulativeHist = ibooker.book2D("track XY profile", title+";x   (mm);y   (mm)", 100, -18., +18., 100, -18., +18.);

  track_u_profile = ibooker.book1D("track profile U", title+"; U   (mm)", 512, -256*66E-3, +256*66E-3);
  track_v_profile = ibooker.book1D("track profile V", title+"; V   (mm)", 512, -256*66E-3, +256*66E-3);
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::PlanePlots::PlanePlots(DQMStore::IBooker &ibooker, unsigned int id)
{
  string path;
  TotemRPDetId(id).planeName(path, TotemRPDetId::nPath);
  ibooker.setCurrentFolder(path);

  string title;
  TotemRPDetId(id).planeName(title, TotemRPDetId::nFull);

  digi_profile_cumulative = ibooker.book1D("digi profile", title+";strip number", 512, -0.5, 511.5);
  cluster_profile_cumulative = ibooker.book1D("cluster profile", title+";cluster center", 1024, -0.25, 511.75);
  hit_multiplicity = ibooker.book1D("hit multiplicity", title+";hits/detector/event", 6, -0.5, 5.5);
  cluster_size = ibooker.book1D("cluster size", title+";hits per cluster", 5, 0.5, 5.5);

  efficiency_num = ibooker.book1D("efficiency num", title+";track position   (mm)", 30, -15., 0.);
  efficiency_den = ibooker.book1D("efficiency den", title+";track position   (mm)", 30, -15., 0.);
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::TotemRPDQMSource(const edm::ParameterSet& ps) :
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0))
{
  tokenStatus = consumes<DetSetVector<TotemVFATStatus>>(ps.getParameter<edm::InputTag>("tagStatus"));

  tokenDigi = consumes< DetSetVector<TotemRPDigi> >(ps.getParameter<edm::InputTag>("tagDigi"));
  tokenCluster = consumes< edm::DetSetVector<TotemRPCluster> >(ps.getParameter<edm::InputTag>("tagCluster"));
  tokenRecHit = consumes< edm::DetSetVector<TotemRPRecHit> >(ps.getParameter<edm::InputTag>("tagRecHit"));
  tokenUVPattern = consumes< DetSetVector<TotemRPUVPattern> >(ps.getParameter<edm::InputTag>("tagUVPattern"));
  tokenLocalTrack = consumes< DetSetVector<TotemRPLocalTrack> >(ps.getParameter<edm::InputTag>("tagLocalTrack"));
  //tokenMultiTrackColl = consumes< RPMulFittedTrackCollection >(ps.getParameter<edm::InputTag>("tagMultiTrackColl"));
}

//----------------------------------------------------------------------------------------------------

TotemRPDQMSource::~TotemRPDQMSource()
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &)
{
  ibooker.cd();
  ibooker.setCurrentFolder("CTPPS");

  // global plots
  globalPlots.Init(ibooker);

  // temporarily disabled
  /*
  // initialize diagonals
  diagonalPlots[1] = DiagonalPlots(ibooker, 1);  // 45 bot - 56 top
  diagonalPlots[2] = DiagonalPlots(ibooker, 2);  // 45 top - 45 bot

  // initialize anti-diagonals
  diagonalPlots[0] = DiagonalPlots(ibooker, 0);  // 45 bot - 56 bot
  diagonalPlots[3] = DiagonalPlots(ibooker, 3);  // 45 top - 56 top
  */

  // loop over arms
  for (unsigned int arm = 0; arm < 2; arm++)
  {
    TotemRPDetId armId(arm, 0);
    armPlots[armId] = ArmPlots(ibooker, armId);

    // loop over stations
    for (unsigned int st = 0; st < 3; st += 2)
    {
      TotemRPDetId stId(arm, st);
      stationPlots[stId] = StationPlots(ibooker, stId);

      // loop over RPs
      for (unsigned int rp = 0; rp < 6; ++rp)
      {
        if (st == 2)
        {
          // unit 220-nr is not equipped
          if (rp <= 2)
            continue;

          // RP 220-fr-hr contains pixels
          if (rp == 3)
            continue;
        }

        TotemRPDetId rpId(arm, st, rp);
        potPlots[rpId] = PotPlots(ibooker, rpId);

        // loop over planes
        for (unsigned int pl = 0; pl < 10; ++pl)
        {
          TotemRPDetId plId(arm, st, rp, pl);
          planePlots[plId] = PlanePlots(ibooker, plId);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) 
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::analyze(edm::Event const& event, edm::EventSetup const& eventSetup)
{
  // get event setup data
  ESHandle<TotemRPGeometry> geometry;
  eventSetup.get<VeryForwardRealGeometryRecord>().get(geometry);

  // get event data
  Handle< DetSetVector<TotemVFATStatus> > status;
  event.getByToken(tokenStatus, status);

  Handle< DetSetVector<TotemRPDigi> > digi;
  event.getByToken(tokenDigi, digi);

  Handle< DetSetVector<TotemRPCluster> > digCluster;
  event.getByToken(tokenCluster, digCluster);

  Handle< DetSetVector<TotemRPRecHit> > hits;
  event.getByToken(tokenRecHit, hits);

  Handle<DetSetVector<TotemRPUVPattern>> patterns;
  event.getByToken(tokenUVPattern, patterns);

  Handle< DetSetVector<TotemRPLocalTrack> > tracks;
  event.getByToken(tokenLocalTrack, tracks);

  //Handle< RPMulFittedTrackCollection > multiTracks;
  //event.getByToken(tokenMultiTrackColl, multiTracks);

  // check validity
  bool valid = true;
  valid &= status.isValid();
  valid &= digi.isValid();
  valid &= digCluster.isValid();
  valid &= hits.isValid();
  valid &= patterns.isValid();
  valid &= tracks.isValid();
  //valid &= multiTracks.isValid();

  if (!valid)
  {
    if (verbosity)
    {
      LogProblem("TotemRPDQMSource") <<
        "ERROR in TotemDQMModuleRP::analyze > some of the required inputs are not valid. Skipping this event.\n"
        << "    status.isValid = " << status.isValid() << "\n"
        << "    digi.isValid = " << digi.isValid() << "\n"
        << "    digCluster.isValid = " << digCluster.isValid() << "\n"
        << "    hits.isValid = " << hits.isValid() << "\n"
        << "    patterns.isValid = " << patterns.isValid() << "\n"
        << "    tracks.isValid = " << tracks.isValid();
      //<< "    multiTracks.isValid = %i\n", multiTracks.isValid()
    }

    return;
  }

  //------------------------------
  // Global Plots

  globalPlots.events_per_bx->Fill(event.bunchCrossing());
  globalPlots.events_per_bx_short->Fill(event.bunchCrossing());

  for (auto &ds1 : *tracks)
  {
    for (auto &tr1 : ds1)
    {
      if (! tr1.isValid())
        continue;
  
      CTPPSDetId rpId1(ds1.detId());
      unsigned int arm1 = rpId1.arm();
      unsigned int stNum1 = rpId1.station();
      unsigned int rpNum1 = rpId1.rp();
      if (stNum1 != 0 || (rpNum1 != 2 && rpNum1 != 3))
        continue;
      unsigned int idx1 = arm1*2 + rpNum1-2;

      for (auto &ds2 : *tracks)
      {
        for (auto &tr2 : ds2)
        {
          if (! tr2.isValid())
            continue;
        
          CTPPSDetId rpId2(ds2.detId());
          unsigned int arm2 = rpId2.arm();
          unsigned int stNum2 = rpId2.station();
          unsigned int rpNum2 = rpId2.rp();
          if (stNum2 != 0 || (rpNum2 != 2 && rpNum2 != 3))
            continue;
          unsigned int idx2 = arm2*2 + rpNum2-2;
  
          globalPlots.h_trackCorr_hor->Fill(idx1, idx2); 
        }
      }
    }
  }

  //------------------------------
  // Status Plots

  for (auto &ds : *status)
  {
    TotemRPDetId detId(ds.detId());
    unsigned int plNum = detId.plane();
    CTPPSDetId rpId = detId.getRPId();

    auto &plots = potPlots[rpId];

    for (auto &s : ds)
    {
      if (s.isMissing())
      {
        plots.vfat_problem->Fill(plNum, s.getChipPosition());
        plots.vfat_missing->Fill(plNum, s.getChipPosition());
      }

      if (s.isECProgressError() || s.isBCProgressError())
      {
        plots.vfat_problem->Fill(plNum, s.getChipPosition());
        plots.vfat_ec_bc_error->Fill(plNum, s.getChipPosition());
      }

      if (s.isIDMismatch() || s.isFootprintError() || s.isCRCError())
      {
        plots.vfat_problem->Fill(plNum, s.getChipPosition());
        plots.vfat_corruption->Fill(plNum, s.getChipPosition());
      }
    }
  }
  
  //------------------------------
  // Plane Plots

  // digi profile cumulative
  for (DetSetVector<TotemRPDigi>::const_iterator it = digi->begin(); it != digi->end(); ++it)
  {
    TotemRPDetId detId(it->detId());
    for (DetSet<TotemRPDigi>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
      planePlots[detId].digi_profile_cumulative->Fill(dit->getStripNumber());
  }

  // cluster profile cumulative
  for (DetSetVector<TotemRPCluster>::const_iterator it = digCluster->begin(); it != digCluster->end(); it++)
  {
    TotemRPDetId detId(it->detId());
    for (DetSet<TotemRPCluster>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
      planePlots[detId].cluster_profile_cumulative->Fill(dit->getCenterStripPosition());
  }

  // hit multiplicity
  for (DetSetVector<TotemRPCluster>::const_iterator it = digCluster->begin(); it != digCluster->end(); it++)
  {
    TotemRPDetId detId(it->detId());
    planePlots[detId].hit_multiplicity->Fill(it->size());
  }

  // cluster size
  for (DetSetVector<TotemRPCluster>::const_iterator it = digCluster->begin(); it != digCluster->end(); it++)
  {
    TotemRPDetId detId(it->detId());
    for (DetSet<TotemRPCluster>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
      planePlots[detId].cluster_size->Fill(dit->getNumberOfStrips());
  }

  // plane efficiency plots
  for (auto &ds : *tracks)
  {
    CTPPSDetId rpId(ds.detId());

    for (auto &ft : ds)
    {
      if (!ft.isValid())
        continue;

      double rp_z = geometry->GetRPGlobalTranslation(rpId).z();

      for (unsigned int plNum = 0; plNum < 10; ++plNum)
      {
        TotemRPDetId plId = rpId;
        plId.setPlane(plNum);

        double ft_z = ft.getZ0();
        double ft_x = ft.getX0() + ft.getTx() * (ft_z - rp_z);
        double ft_y = ft.getY0() + ft.getTy() * (ft_z - rp_z);

        double ft_v = geometry->GlobalToLocal(plId, CLHEP::Hep3Vector(ft_x, ft_y, ft_z)).y();

        bool hasMatchingHit = false;
        const auto &hit_ds_it = hits->find(plId);
        if (hit_ds_it != hits->end())
        {
          for (const auto &h : *hit_ds_it)
          {
            bool match = (fabs(ft_v - h.getPosition()) < 2.*0.066);
            if (match)
            {
              hasMatchingHit = true;
              break;
            }
          }
        }

        auto &pp = planePlots[plId];

        pp.efficiency_den->Fill(ft_v);
        if (hasMatchingHit)
          pp.efficiency_num->Fill(ft_v);
      }
    }
  }


  //------------------------------
  // Roman Pots Plots

  // determine active planes (from RecHits and VFATStatus)
  map<unsigned int, set<unsigned int> > planes;
  map<unsigned int, set<unsigned int> > planes_u;
  map<unsigned int, set<unsigned int> > planes_v;
  for (const auto &ds : *hits)
  {
    if (ds.empty())
      continue;

    TotemRPDetId detId(ds.detId());
    unsigned int planeNum = detId.plane();
    CTPPSDetId rpId = detId.getRPId();

    planes[rpId].insert(planeNum);
    if (detId.isStripsCoordinateUDirection())
      planes_u[rpId].insert(planeNum);
    else
      planes_v[rpId].insert(planeNum);
  }

  for (const auto &ds : *status)
  {
    bool activity = false;
    for (const auto &s : ds)
    {
      if (s.isNumberOfClustersSpecified() && s.getNumberOfClusters() > 0)
      {
        activity = true;
        break;
      }
    } 

    if (!activity)
      continue;

    TotemRPDetId detId(ds.detId());
    unsigned int planeNum = detId.plane();
    CTPPSDetId rpId = detId.getRPId();

    planes[rpId].insert(planeNum);
    if (detId.isStripsCoordinateUDirection())
      planes_u[rpId].insert(planeNum);
    else
      planes_v[rpId].insert(planeNum);
  }

  // plane activity histogram
  for (std::map<unsigned int, PotPlots>::iterator it = potPlots.begin(); it != potPlots.end(); it++)
  {
    it->second.activity->Fill(planes[it->first].size());
    it->second.activity_u->Fill(planes_u[it->first].size());
    it->second.activity_v->Fill(planes_v[it->first].size());

    if (planes[it->first].size() >= 6)
    {
      it->second.activity_per_bx->Fill(event.bunchCrossing());
      it->second.activity_per_bx_short->Fill(event.bunchCrossing());
    }
  }
  
  for (DetSetVector<TotemRPCluster>::const_iterator it = digCluster->begin(); it != digCluster->end(); it++)
  {
    TotemRPDetId detId(it->detId());
    unsigned int planeNum = detId.plane();
    CTPPSDetId rpId = detId.getRPId();

    PotPlots &pp = potPlots[rpId];
    for (DetSet<TotemRPCluster>::const_iterator dit = it->begin(); dit != it->end(); ++dit)
      pp.hit_plane_hist->Fill(planeNum, dit->getCenterStripPosition());   
  }

  // recognized pattern histograms
  for (auto &ds : *patterns)
  {
    CTPPSDetId rpId(ds.detId());

    PotPlots &pp = potPlots[rpId];

    // count U and V patterns
    unsigned int u = 0, v = 0;
    for (auto &p : ds)
    {
      if (! p.getFittable())
        continue;

      if (p.getProjection() == TotemRPUVPattern::projU)
        u++;

      if (p.getProjection() == TotemRPUVPattern::projV)
        v++;
    }

    pp.patterns_u->Fill(u);
    pp.patterns_v->Fill(v);
  }

  // event-category histogram
  for (auto &it : potPlots)
  {
    TotemRPDetId rpId(it.first);
    auto &pp = it.second;

    // process hit data for this plot
    unsigned int pl_u = planes_u[rpId].size();
    unsigned int pl_v = planes_v[rpId].size();

    // process pattern data for this pot
    const auto &rp_pat_it = patterns->find(rpId);

    unsigned int pat_u = 0, pat_v = 0;
    if (rp_pat_it != patterns->end())
    {
      for (auto &p : *rp_pat_it)
      {
        if (! p.getFittable())
          continue;
  
        if (p.getProjection() == TotemRPUVPattern::projU)
          pat_u++;
  
        if (p.getProjection() == TotemRPUVPattern::projV)
          pat_v++;
      }
    }

    // determine category
    signed int category = -1;

    if (pl_u == 0 && pl_v == 0) category = 0;   // empty
    
    if (category == -1 && pat_u + pat_v <= 1)
    {
      if (pl_u + pl_v < 6)
        category = 1;                           // insuff
      else
        category = 4;                           // shower
    }

    if (pat_u == 1 && pat_v == 1) category = 2; // 1-track

    if (category == -1) category = 3;           // multi-track

    pp.event_category->Fill(category);
  }

  // RP track-fit plots
  for (auto &ds : *tracks)
  {
    CTPPSDetId rpId(ds.detId());

    PotPlots &pp = potPlots[rpId];

    for (auto &ft : ds)
    {
      if (!ft.isValid())
        continue;
     
      // number of planes contributing to (valid) fits
      unsigned int n_pl_in_fit_u = 0, n_pl_in_fit_v = 0;
      for (auto &hds : ft.getHits())
      {
        TotemRPDetId plId(hds.detId());
        bool uProj = plId.isStripsCoordinateUDirection();

        for (auto &h : hds)
        {
          h.getPosition();  // just to keep compiler silent
          if (uProj)
            n_pl_in_fit_u++;
          else
            n_pl_in_fit_v++;
        }
      }

      pp.h_planes_fit_u->Fill(n_pl_in_fit_u);
      pp.h_planes_fit_v->Fill(n_pl_in_fit_v);
  
      // mean position of U and V planes
      TotemRPDetId plId_V(rpId); plId_V.setPlane(0);
      TotemRPDetId plId_U(rpId); plId_U.setPlane(1);

      double rp_x = ( geometry->GetDetector(plId_V)->translation().x() +
                      geometry->GetDetector(plId_U)->translation().x() ) / 2.;
      double rp_y = ( geometry->GetDetector(plId_V)->translation().y() +
                      geometry->GetDetector(plId_U)->translation().y() ) / 2.;
  
      // mean read-out direction of U and V planes
      CLHEP::Hep3Vector rod_U = geometry->LocalToGlobalDirection(plId_U, CLHEP::Hep3Vector(0., 1., 0.));
      CLHEP::Hep3Vector rod_V = geometry->LocalToGlobalDirection(plId_V, CLHEP::Hep3Vector(0., 1., 0.));
  
      double x = ft.getX0() - rp_x;
      double y = ft.getY0() - rp_y;
  
      pp.trackHitsCumulativeHist->Fill(x, y);
  
      double U = x * rod_U.x() + y * rod_U.y();
      double V = x * rod_V.x() + y * rod_V.y();
  
      pp.track_u_profile->Fill(U);
      pp.track_v_profile->Fill(V);
    }
  }

  //------------------------------
  // Station Plots

  
  //------------------------------
  // Arm Plots
  {
    map<unsigned int, unsigned int> mTop, mHor, mBot;

    for (auto p : armPlots)
    {
      mTop[p.first] = 0;
      mHor[p.first] = 0;
      mBot[p.first] = 0;
    }

    for (auto &ds : *tracks)
    {
      CTPPSDetId rpId(ds.detId());
      unsigned int rpNum = rpId.rp();
      CTPPSDetId armId = rpId.getArmId();

      for (auto &tr : ds)
      {
        if (! tr.isValid())
          continue;
  
        if (rpNum == 0 || rpNum == 4)
          mTop[armId]++;
        if (rpNum == 2 || rpNum == 3)
          mHor[armId]++;
        if (rpNum == 1 || rpNum == 5)
          mBot[armId]++;
      }
    }

    for (auto &p : armPlots)
    {
      p.second.h_numRPWithTrack_top->Fill(mTop[p.first]);
      p.second.h_numRPWithTrack_hor->Fill(mHor[p.first]);
      p.second.h_numRPWithTrack_bot->Fill(mBot[p.first]);
    }

    // track RP correlation
    for (auto &ds1 : *tracks)
    {
      for (auto &tr1 : ds1)
      {
        if (! tr1.isValid())
          continue;
  
        CTPPSDetId rpId1(ds1.detId());
        unsigned int arm1 = rpId1.arm();
        unsigned int stNum1 = rpId1.station();
        unsigned int rpNum1 = rpId1.rp();
        unsigned int idx1 = stNum1/2 * 7 + rpNum1;
        bool hor1 = (rpNum1 == 2 || rpNum1 == 3);
  
        CTPPSDetId armId = rpId1.getArmId();
        ArmPlots &ap = armPlots[armId];
  
        for (auto &ds2 : *tracks)
        {
          for (auto &tr2 : ds2)
          {
            if (! tr2.isValid())
              continue;
          
            CTPPSDetId rpId2(ds2.detId());
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
  }
  
  //------------------------------
  // RP-system plots
  // TODO: this code needs
  //    * generalization for more than two RPs per arm
  //    * updating for tracks as DetSetVector
  /*
  for (auto &dp : diagonalPlots)
  {
    unsigned int id = dp.first;
    bool top45 = id & 2;
    bool top56 = id & 1;

    unsigned int id_45_n = (top45) ? 20 : 21;
    unsigned int id_45_f = (top45) ? 24 : 25;
    unsigned int id_56_n = (top56) ? 120 : 121;
    unsigned int id_56_f = (top56) ? 124 : 125;
  
    bool h_45_n = (tracks->find(id_45_n) != tracks->end() && tracks->find(id_45_n)->second.IsValid());
    bool h_45_f = (tracks->find(id_45_f) != tracks->end() && tracks->find(id_45_f)->second.IsValid());
    bool h_56_n = (tracks->find(id_56_n) != tracks->end() && tracks->find(id_56_n)->second.IsValid());
    bool h_56_f = (tracks->find(id_56_f) != tracks->end() && tracks->find(id_56_f)->second.IsValid());
  
    if (! (h_45_n && h_45_f && h_56_n && h_56_f) )
      continue;

    double x_45_n = tracks->find(id_45_n)->second.X0(), y_45_n = tracks->find(id_45_n)->second.Y0();
    double x_45_f = tracks->find(id_45_f)->second.X0(), y_45_f = tracks->find(id_45_f)->second.Y0();
    double x_56_n = tracks->find(id_56_n)->second.X0(), y_56_n = tracks->find(id_56_n)->second.Y0();
    double x_56_f = tracks->find(id_56_f)->second.X0(), y_56_f = tracks->find(id_56_f)->second.Y0();

    double dx_45 = x_45_f - x_45_n;
    double dy_45 = y_45_f - y_45_n;
    double dx_56 = x_56_f - x_56_n;
    double dy_56 = y_56_f - y_56_n;

    DiagonalPlots &pl = dp.second;

    pl.h_lrc_x_d->Fill(dx_45, dx_56);  
    pl.h_lrc_y_d->Fill(dy_45, dy_56);  
    
    pl.h_lrc_x_n->Fill(x_45_n, x_56_n);  
    pl.h_lrc_y_n->Fill(y_45_n, y_56_n);  
    
    pl.h_lrc_x_f->Fill(x_45_f, x_56_f);  
    pl.h_lrc_y_f->Fill(y_45_f, y_56_f);
  }
  */
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) 
{
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMSource::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRPDQMSource);
