/*
 * \file FakeBeamMonitor.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 */

/*
The code has been modified for running average
mode, and it gives results for the last NLS which is
configurable.
Sushil S. Chauhan /UCDavis
Evan Friis        /UCDavis
The last tag for working versions without this change is
V00-03-25
*/

#include "DQM/BeamMonitor/plugins/FakeBeamMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Common/interface/View.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include <numeric>
#include <cmath>
#include <memory>
#include <TMath.h>
#include <TDatime.h>
#include <iostream>
#include <TStyle.h>
#include <ctime>

using namespace std;
using namespace edm;

void FakeBeamMonitor::formatFitTime(char* ts, const time_t& t) {
  //constexpr int CET(+1);
  constexpr int CEST(+2);

  //tm * ptm;
  //ptm = gmtime ( &t );
  //int year = ptm->tm_year;

  //get correct year from ctime
  time_t currentTime;
  struct tm* localTime;
  time(&currentTime);                   // Get the current time
  localTime = localtime(&currentTime);  // Convert the current time to the local time
  int year = localTime->tm_year + 1900;

  tm* ptm;
  ptm = gmtime(&t);

  //check if year is ok
  if (year <= 37)
    year += 2000;
  if (year >= 70 && year <= 137)
    year += 1900;

  if (year < 1995) {
    edm::LogError("BadTimeStamp") << "year reported is " << year << " !!" << std::endl;
    //year = 2015; //overwritten later by BeamFitter.cc for fits but needed here for TH1
    //edm::LogError("BadTimeStamp") << "Resetting to " <<year<<std::endl;
  }
  sprintf(ts,
          "%4d-%02d-%02d %02d:%02d:%02d",
          year,
          ptm->tm_mon + 1,
          ptm->tm_mday,
          (ptm->tm_hour + CEST) % 24,
          ptm->tm_min,
          ptm->tm_sec);

#ifdef STRIP_TRAILING_BLANKS_IN_TIMEZONE
  unsigned int b = strlen(ts);
  while (ts[--b] == ' ') {
    ts[b] = 0;
  }
#endif
}

static constexpr int buffTime = 23;

//
// constructors and destructor
//
FakeBeamMonitor::~FakeBeamMonitor() { delete rndm_; };

FakeBeamMonitor::FakeBeamMonitor(const ParameterSet& ps)
    : dxBin_(ps.getParameter<int>("dxBin")),
      dxMin_(ps.getParameter<double>("dxMin")),
      dxMax_(ps.getParameter<double>("dxMax")),

      vxBin_(ps.getParameter<int>("vxBin")),
      vxMin_(ps.getParameter<double>("vxMin")),
      vxMax_(ps.getParameter<double>("vxMax")),

      phiBin_(ps.getParameter<int>("phiBin")),
      phiMin_(ps.getParameter<double>("phiMin")),
      phiMax_(ps.getParameter<double>("phiMax")),

      dzBin_(ps.getParameter<int>("dzBin")),
      dzMin_(ps.getParameter<double>("dzMin")),
      dzMax_(ps.getParameter<double>("dzMax")),

      countEvt_(0),
      countLumi_(0),
      nthBSTrk_(0),
      nFitElements_(3),
      resetHistos_(false),
      StartAverage_(false),
      firstAverageFit_(0),
      countGapLumi_(0) {
  monitorName_ = ps.getUntrackedParameter<string>("monitorName", "YourSubsystemName");
  recordName_ = ps.getUntrackedParameter<string>("recordName");
  intervalInSec_ = ps.getUntrackedParameter<int>("timeInterval", 920);  //40 LS X 23"
  fitNLumi_ = ps.getUntrackedParameter<int>("fitEveryNLumi", -1);
  resetFitNLumi_ = ps.getUntrackedParameter<int>("resetEveryNLumi", -1);
  fitPVNLumi_ = ps.getUntrackedParameter<int>("fitPVEveryNLumi", -1);
  resetPVNLumi_ = ps.getUntrackedParameter<int>("resetPVEveryNLumi", -1);
  deltaSigCut_ = ps.getUntrackedParameter<double>("deltaSignificanceCut", 15);
  debug_ = ps.getUntrackedParameter<bool>("Debug");
  onlineMode_ = ps.getUntrackedParameter<bool>("OnlineMode");
  min_Ntrks_ = ps.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumInputTracks");
  maxZ_ = ps.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumZ");
  minNrVertices_ = ps.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<unsigned int>("minNrVerticesForFit");
  minVtxNdf_ = ps.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexNdf");
  minVtxWgt_ = ps.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexMeanWeight");

  if (!monitorName_.empty())
    monitorName_ = monitorName_ + "/";

  if (fitNLumi_ <= 0)
    fitNLumi_ = 1;
  nFits_ = beginLumiOfBSFit_ = endLumiOfBSFit_ = beginLumiOfPVFit_ = endLumiOfPVFit_ = 0;
  refBStime[0] = refBStime[1] = refPVtime[0] = refPVtime[1] = 0;
  maxZ_ = std::fabs(maxZ_);
  lastlumi_ = 0;
  nextlumi_ = 0;
  processed_ = false;

  rndm_ = new TRandom3(0);
}

//--------------------------------------------------------
namespace {
  /*The order of the enums is identical to the order in which
    MonitorElements are added to hs
   */
  enum Hists {
    k_x0_lumi,
    k_x0_lumi_all,
    k_y0_lumi,
    k_y0_lumi_all,
    k_z0_lumi,
    k_z0_lumi_all,
    k_sigmaX0_lumi,
    k_sigmaX0_lumi_all,
    k_sigmaY0_lumi,
    k_sigmaY0_lumi_all,
    k_sigmaZ0_lumi,
    k_sigmaZ0_lumi_all,
    k_x0_time,
    k_x0_time_all,
    k_y0_time,
    k_y0_time_all,
    k_z0_time,
    k_z0_time_all,
    k_sigmaX0_time,
    k_sigmaX0_time_all,
    k_sigmaY0_time,
    k_sigmaY0_time_all,
    k_sigmaZ0_time,
    k_sigmaZ0_time_all,
    k_PVx_lumi,
    k_PVx_lumi_all,
    k_PVy_lumi,
    k_PVy_lumi_all,
    k_PVz_lumi,
    k_PVz_lumi_all,
    k_PVx_time,
    k_PVx_time_all,
    k_PVy_time,
    k_PVy_time_all,
    k_PVz_time,
    k_PVz_time_all,
    kNumHists
  };
}  // namespace

void FakeBeamMonitor::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  frun = iRun.run();
  ftimestamp = iRun.beginTime().value();
  tmpTime = ftimestamp >> 32;
  startTime = refTime = tmpTime;
  char eventTime[64];
  formatFitTime(eventTime, tmpTime);
  edm::LogInfo("FakeBeamMonitor") << "TimeOffset = " << eventTime << std::endl;
  TDatime da(eventTime);
  if (debug_) {
    edm::LogInfo("FakeBeamMonitor") << "TimeOffset = ";
    da.Print();
  }
  auto daTime = da.Convert(kTRUE);

  // book some histograms here

  // create and cd into new folder
  iBooker.setCurrentFolder(monitorName_ + "Fit");

  h_nTrk_lumi = iBooker.book1D("nTrk_lumi", "Num. of selected tracks vs lumi (Fit)", 20, 0.5, 20.5);
  h_nTrk_lumi->setAxisTitle("Lumisection", 1);
  h_nTrk_lumi->setAxisTitle("Num of Tracks for Fit", 2);

  //store vtx vs lumi for monitoring why fits fail
  h_nVtx_lumi = iBooker.book1D("nVtx_lumi", "Num. of selected Vtx vs lumi (Fit)", 20, 0.5, 20.5);
  h_nVtx_lumi->setAxisTitle("Lumisection", 1);
  h_nVtx_lumi->setAxisTitle("Num of Vtx for Fit", 2);

  h_nVtx_lumi_all = iBooker.book1D("nVtx_lumi_all", "Num. of selected Vtx vs lumi (Fit) all", 20, 0.5, 20.5);
  h_nVtx_lumi_all->getTH1()->SetCanExtend(TH1::kAllAxes);
  h_nVtx_lumi_all->setAxisTitle("Lumisection", 1);
  h_nVtx_lumi_all->setAxisTitle("Num of Vtx for Fit", 2);

  h_d0_phi0 = iBooker.bookProfile(
      "d0_phi0", "d_{0} vs. #phi_{0} (Selected Tracks)", phiBin_, phiMin_, phiMax_, dxBin_, dxMin_, dxMax_, "");
  h_d0_phi0->setAxisTitle("#phi_{0} (rad)", 1);
  h_d0_phi0->setAxisTitle("d_{0} (cm)", 2);

  h_vx_vy = iBooker.book2D(
      "trk_vx_vy", "Vertex (PCA) position of selected tracks", vxBin_, vxMin_, vxMax_, vxBin_, vxMin_, vxMax_);
  h_vx_vy->setOption("COLZ");
  //   h_vx_vy->getTH1()->SetCanExtend(TH1::kAllAxes);
  h_vx_vy->setAxisTitle("x coordinate of input track at PCA (cm)", 1);
  h_vx_vy->setAxisTitle("y coordinate of input track at PCA (cm)", 2);

  {
    TDatime* da = new TDatime();
    gStyle->SetTimeOffset(da->Convert(kTRUE));
  }

  const int nvar_ = 6;
  string coord[nvar_] = {"x", "y", "z", "sigmaX", "sigmaY", "sigmaZ"};
  string label[nvar_] = {
      "x_{0} (cm)", "y_{0} (cm)", "z_{0} (cm)", "#sigma_{X_{0}} (cm)", "#sigma_{Y_{0}} (cm)", "#sigma_{Z_{0}} (cm)"};

  hs.clear();
  hs.reserve(kNumHists);
  for (int i = 0; i < 4; i++) {
    iBooker.setCurrentFolder(monitorName_ + "Fit");
    for (int ic = 0; ic < nvar_; ++ic) {
      TString histName(coord[ic]);
      TString histTitle(coord[ic]);
      string ytitle(label[ic]);
      string xtitle("");
      string options("E1");
      bool createHisto = true;
      switch (i) {
        case 1:  // BS vs time
          histName += "0_time";
          xtitle = "Time [UTC]";
          if (ic < 3)
            histTitle += " coordinate of beam spot vs time (Fit)";
          else
            histTitle = histTitle.Insert(5, " ") + " of beam spot vs time (Fit)";
          break;
        case 2:  // PV vs lumi
          if (ic < 3) {
            iBooker.setCurrentFolder(monitorName_ + "PrimaryVertex");
            histName.Insert(0, "PV");
            histName += "_lumi";
            histTitle.Insert(0, "Avg. ");
            histTitle += " position of primary vtx vs lumi";
            xtitle = "Lumisection";
            ytitle.insert(0, "PV");
            ytitle += " #pm #sigma_{PV";
            ytitle += coord[ic];
            ytitle += "} (cm)";
          } else
            createHisto = false;
          break;
        case 3:  // PV vs time
          if (ic < 3) {
            iBooker.setCurrentFolder(monitorName_ + "PrimaryVertex");
            histName.Insert(0, "PV");
            histName += "_time";
            histTitle.Insert(0, "Avg. ");
            histTitle += " position of primary vtx vs time";
            xtitle = "Time [UTC]";
            ytitle.insert(0, "PV");
            ytitle += " #pm #sigma_{PV";
            ytitle += coord[ic];
            ytitle += "} (cm)";
          } else
            createHisto = false;
          break;
        default:  // BS vs lumi
          histName += "0_lumi";
          xtitle = "Lumisection";
          if (ic < 3)
            histTitle += " coordinate of beam spot vs lumi (Fit)";
          else
            histTitle = histTitle.Insert(5, " ") + " of beam spot vs lumi (Fit)";
          break;
      }
      if (createHisto) {
        edm::LogInfo("FakeBeamMonitor") << "hitsName = " << histName << "; histTitle = " << histTitle << std::endl;
        auto tmpHs = iBooker.book1D(histName, histTitle, 40, 0.5, 40.5);
        hs.push_back(tmpHs);
        tmpHs->setAxisTitle(xtitle, 1);
        tmpHs->setAxisTitle(ytitle, 2);
        tmpHs->getTH1()->SetOption("E1");
        if (histName.Contains("time")) {
          //int nbins = (intervalInSec_/23 > 0 ? intervalInSec_/23 : 40);
          tmpHs->getTH1()->SetBins(intervalInSec_, 0.5, intervalInSec_ + 0.5);
          tmpHs->setAxisTimeDisplay(1);
          tmpHs->setAxisTimeFormat("%H:%M:%S", 1);
          tmpHs->getTH1()->GetXaxis()->SetTimeOffset(daTime);
        }
        histName += "_all";
        histTitle += " all";
        tmpHs = iBooker.book1D(histName, histTitle, 40, 0.5, 40.5);
        hs.push_back(tmpHs);
        tmpHs->getTH1()->SetCanExtend(TH1::kAllAxes);
        tmpHs->setAxisTitle(xtitle, 1);
        tmpHs->setAxisTitle(ytitle, 2);
        tmpHs->getTH1()->SetOption("E1");
        if (histName.Contains("time")) {
          //int nbins = (intervalInSec_/23 > 0 ? intervalInSec_/23 : 40);
          tmpHs->getTH1()->SetBins(intervalInSec_, 0.5, intervalInSec_ + 0.5);
          tmpHs->setAxisTimeDisplay(1);
          tmpHs->setAxisTimeFormat("%H:%M:%S", 1);
          tmpHs->getTH1()->GetXaxis()->SetTimeOffset(daTime);
        }
      }
    }
  }
  assert(hs.size() == kNumHists);
  assert(0 == strcmp(hs[k_sigmaY0_time]->getTH1()->GetName(), "sigmaY0_time"));
  assert(0 == strcmp(hs[k_PVz_lumi_all]->getTH1()->GetName(), "PVz_lumi_all"));

  iBooker.setCurrentFolder(monitorName_ + "Fit");

  h_trk_z0 = iBooker.book1D("trk_z0", "z_{0} of selected tracks", dzBin_, dzMin_, dzMax_);
  h_trk_z0->setAxisTitle("z_{0} of selected tracks (cm)", 1);

  h_vx_dz = iBooker.bookProfile(
      "vx_dz", "v_{x} vs. dz of selected tracks", dzBin_, dzMin_, dzMax_, dxBin_, dxMin_, dxMax_, "");
  h_vx_dz->setAxisTitle("dz (cm)", 1);
  h_vx_dz->setAxisTitle("x coordinate of input track at PCA (cm)", 2);

  h_vy_dz = iBooker.bookProfile(
      "vy_dz", "v_{y} vs. dz of selected tracks", dzBin_, dzMin_, dzMax_, dxBin_, dxMin_, dxMax_, "");
  h_vy_dz->setAxisTitle("dz (cm)", 1);
  h_vy_dz->setAxisTitle("y coordinate of input track at PCA (cm)", 2);

  h_x0 = iBooker.book1D("BeamMonitorFeedBack_x0", "x coordinate of beam spot (Fit)", 100, -0.01, 0.01);
  h_x0->setAxisTitle("x_{0} (cm)", 1);
  h_x0->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_y0 = iBooker.book1D("BeamMonitorFeedBack_y0", "y coordinate of beam spot (Fit)", 100, -0.01, 0.01);
  h_y0->setAxisTitle("y_{0} (cm)", 1);
  h_y0->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_z0 = iBooker.book1D("BeamMonitorFeedBack_z0", "z coordinate of beam spot (Fit)", dzBin_, dzMin_, dzMax_);
  h_z0->setAxisTitle("z_{0} (cm)", 1);
  h_z0->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_sigmaX0 = iBooker.book1D("BeamMonitorFeedBack_sigmaX0", "sigma x0 of beam spot (Fit)", 100, 0, 0.05);
  h_sigmaX0->setAxisTitle("#sigma_{X_{0}} (cm)", 1);
  h_sigmaX0->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_sigmaY0 = iBooker.book1D("BeamMonitorFeedBack_sigmaY0", "sigma y0 of beam spot (Fit)", 100, 0, 0.05);
  h_sigmaY0->setAxisTitle("#sigma_{Y_{0}} (cm)", 1);
  h_sigmaY0->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_sigmaZ0 = iBooker.book1D("BeamMonitorFeedBack_sigmaZ0", "sigma z0 of beam spot (Fit)", 100, 0, 10);
  h_sigmaZ0->setAxisTitle("#sigma_{Z_{0}} (cm)", 1);
  h_sigmaZ0->getTH1()->SetCanExtend(TH1::kAllAxes);

  // Histograms of all reco tracks (without cuts):
  h_trkPt = iBooker.book1D("trkPt", "p_{T} of all reco'd tracks (no selection)", 200, 0., 50.);
  h_trkPt->setAxisTitle("p_{T} (GeV/c)", 1);

  h_trkVz = iBooker.book1D("trkVz", "Z coordinate of PCA of all reco'd tracks (no selection)", dzBin_, dzMin_, dzMax_);
  h_trkVz->setAxisTitle("V_{Z} (cm)", 1);

  cutFlowTable = iBooker.book1D("cutFlowTable", "Cut flow table of track selection", 9, 0, 9);

  // Results of previous good fit:
  fitResults = iBooker.book2D("fitResults", "Results of previous good beam fit", 2, 0, 2, 8, 0, 8);
  fitResults->setAxisTitle("Fitted Beam Spot (cm)", 1);
  fitResults->setBinLabel(8, "x_{0}", 2);
  fitResults->setBinLabel(7, "y_{0}", 2);
  fitResults->setBinLabel(6, "z_{0}", 2);
  fitResults->setBinLabel(5, "#sigma_{Z}", 2);
  fitResults->setBinLabel(4, "#frac{dx}{dz} (rad)", 2);
  fitResults->setBinLabel(3, "#frac{dy}{dz} (rad)", 2);
  fitResults->setBinLabel(2, "#sigma_{X}", 2);
  fitResults->setBinLabel(1, "#sigma_{Y}", 2);
  fitResults->setBinLabel(1, "Mean", 1);
  fitResults->setBinLabel(2, "Stat. Error", 1);
  fitResults->getTH1()->SetOption("text");

  // Histos of PrimaryVertices:
  iBooker.setCurrentFolder(monitorName_ + "PrimaryVertex");

  h_nVtx = iBooker.book1D("vtxNbr", "Reconstructed Vertices(non-fake) in all Event", 60, -0.5, 59.5);
  h_nVtx->setAxisTitle("Num. of reco. vertices", 1);

  //For one Trigger only
  h_nVtx_st = iBooker.book1D("vtxNbr_SelectedTriggers", "Reconstructed Vertices(non-fake) in Events", 60, -0.5, 59.5);
  //h_nVtx_st->setAxisTitle("Num. of reco. vertices for Un-Prescaled Jet Trigger",1);

  // Monitor only the PV with highest sum pt of assoc. trks:
  h_PVx[0] = iBooker.book1D("PVX", "x coordinate of Primary Vtx", 50, -0.01, 0.01);
  h_PVx[0]->setAxisTitle("PVx (cm)", 1);
  h_PVx[0]->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_PVy[0] = iBooker.book1D("PVY", "y coordinate of Primary Vtx", 50, -0.01, 0.01);
  h_PVy[0]->setAxisTitle("PVy (cm)", 1);
  h_PVy[0]->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_PVz[0] = iBooker.book1D("PVZ", "z coordinate of Primary Vtx", dzBin_, dzMin_, dzMax_);
  h_PVz[0]->setAxisTitle("PVz (cm)", 1);

  h_PVx[1] = iBooker.book1D("PVXFit", "x coordinate of Primary Vtx (Last Fit)", 50, -0.01, 0.01);
  h_PVx[1]->setAxisTitle("PVx (cm)", 1);
  h_PVx[1]->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_PVy[1] = iBooker.book1D("PVYFit", "y coordinate of Primary Vtx (Last Fit)", 50, -0.01, 0.01);
  h_PVy[1]->setAxisTitle("PVy (cm)", 1);
  h_PVy[1]->getTH1()->SetCanExtend(TH1::kAllAxes);

  h_PVz[1] = iBooker.book1D("PVZFit", "z coordinate of Primary Vtx (Last Fit)", dzBin_, dzMin_, dzMax_);
  h_PVz[1]->setAxisTitle("PVz (cm)", 1);

  h_PVxz = iBooker.bookProfile("PVxz", "PVx vs. PVz", dzBin_ / 2, dzMin_, dzMax_, dxBin_ / 2, dxMin_, dxMax_, "");
  h_PVxz->setAxisTitle("PVz (cm)", 1);
  h_PVxz->setAxisTitle("PVx (cm)", 2);

  h_PVyz = iBooker.bookProfile("PVyz", "PVy vs. PVz", dzBin_ / 2, dzMin_, dzMax_, dxBin_ / 2, dxMin_, dxMax_, "");
  h_PVyz->setAxisTitle("PVz (cm)", 1);
  h_PVyz->setAxisTitle("PVy (cm)", 2);

  // Results of previous good fit:
  pvResults = iBooker.book2D("pvResults", "Results of fitting Primary Vertices", 2, 0, 2, 6, 0, 6);
  pvResults->setAxisTitle("Fitted Primary Vertex (cm)", 1);
  pvResults->setBinLabel(6, "PVx", 2);
  pvResults->setBinLabel(5, "PVy", 2);
  pvResults->setBinLabel(4, "PVz", 2);
  pvResults->setBinLabel(3, "#sigma_{X}", 2);
  pvResults->setBinLabel(2, "#sigma_{Y}", 2);
  pvResults->setBinLabel(1, "#sigma_{Z}", 2);
  pvResults->setBinLabel(1, "Mean", 1);
  pvResults->setBinLabel(2, "Stat. Error", 1);
  pvResults->getTH1()->SetOption("text");

  // Summary plots:
  iBooker.setCurrentFolder(monitorName_ + "EventInfo");

  reportSummary = iBooker.bookFloat("reportSummary");
  if (reportSummary)
    reportSummary->Fill(std::numeric_limits<double>::quiet_NaN());

  char histo[20];
  iBooker.setCurrentFolder(monitorName_ + "EventInfo/reportSummaryContents");
  for (int n = 0; n < nFitElements_; n++) {
    switch (n) {
      case 0:
        sprintf(histo, "x0_status");
        break;
      case 1:
        sprintf(histo, "y0_status");
        break;
      case 2:
        sprintf(histo, "z0_status");
        break;
    }
    reportSummaryContents[n] = iBooker.bookFloat(histo);
  }

  for (int i = 0; i < nFitElements_; i++) {
    summaryContent_[i] = 0.;
    reportSummaryContents[i]->Fill(std::numeric_limits<double>::quiet_NaN());
  }

  iBooker.setCurrentFolder(monitorName_ + "EventInfo");

  reportSummaryMap = iBooker.book2D("reportSummaryMap", "Beam Spot Summary Map", 1, 0, 1, 3, 0, 3);
  reportSummaryMap->setAxisTitle("", 1);
  reportSummaryMap->setAxisTitle("Fitted Beam Spot", 2);
  reportSummaryMap->setBinLabel(1, " ", 1);
  reportSummaryMap->setBinLabel(1, "x_{0}", 2);
  reportSummaryMap->setBinLabel(2, "y_{0}", 2);
  reportSummaryMap->setBinLabel(3, "z_{0}", 2);
  for (int i = 0; i < nFitElements_; i++) {
    reportSummaryMap->setBinContent(1, i + 1, -1.);
  }
}

//--------------------------------------------------------
void FakeBeamMonitor::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  // start DB logger
  DBloggerReturn_ = 0;
  if (onlineDbService_.isAvailable()) {
    onlineDbService_->logger().start();
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor::beginLuminosityBlock";
  }

  int nthlumi = lumiSeg.luminosityBlock();
  const edm::TimeValue_t fbegintimestamp = lumiSeg.beginTime().value();
  const std::time_t ftmptime = fbegintimestamp >> 32;

  if (countLumi_ == 0 && (!processed_)) {
    beginLumiOfBSFit_ = beginLumiOfPVFit_ = nthlumi;
    refBStime[0] = refPVtime[0] = ftmptime;
    mapBeginBSLS[countLumi_] = nthlumi;
    mapBeginPVLS[countLumi_] = nthlumi;
    mapBeginBSTime[countLumi_] = ftmptime;
    mapBeginPVTime[countLumi_] = ftmptime;
  }  //for the first record

  if (nthlumi > nextlumi_) {
    if (processed_) {
      countLumi_++;
      //store here them will need when we remove the first one of Last N LS
      mapBeginBSLS[countLumi_] = nthlumi;
      mapBeginPVLS[countLumi_] = nthlumi;
      mapBeginBSTime[countLumi_] = ftmptime;
      mapBeginPVTime[countLumi_] = ftmptime;
    }  //processed passed but not the first lumi
    if ((!processed_) && countLumi_ != 0) {
      mapBeginBSLS[countLumi_] = nthlumi;
      mapBeginPVLS[countLumi_] = nthlumi;
      mapBeginBSTime[countLumi_] = ftmptime;
      mapBeginPVTime[countLumi_] = ftmptime;
    }  //processed fails for last lumi
  }    //nthLumi > nextlumi

  if (StartAverage_) {
    //Just Make sure it get rest
    refBStime[0] = 0;
    refPVtime[0] = 0;
    beginLumiOfPVFit_ = 0;
    beginLumiOfBSFit_ = 0;

    if (debug_)
      edm::LogInfo("FakeBeamMonitor") << " beginLuminosityBlock:  Size of mapBeginBSLS before =  "
                                      << mapBeginBSLS.size() << endl;
    if (nthlumi >
        nextlumi_) {  //this make sure that it does not take into account this lumi for fitting and only look forward for new lumi
      //as countLumi also remains the same so map value  get overwritten once return to normal running.
      //even if few LS are misssing and DQM module do not sees them then it catchs up again
      map<int, int>::iterator itbs = mapBeginBSLS.begin();
      map<int, int>::iterator itpv = mapBeginPVLS.begin();
      map<int, std::time_t>::iterator itbstime = mapBeginBSTime.begin();
      map<int, std::time_t>::iterator itpvtime = mapBeginPVTime.begin();

      if (processed_) {  // otherwise if false then LS range of fit get messed up because we don't remove trk/pvs but we remove LS begin value . This prevent it as it happened if LS is there but no event are processed for some reason
        mapBeginBSLS.erase(itbs);
        mapBeginPVLS.erase(itpv);
        mapBeginBSTime.erase(itbstime);
        mapBeginPVTime.erase(itpvtime);
      }
      /*//not sure if want this or not ??
            map<int, int>::iterator itgapb=mapBeginBSLS.begin();
            map<int, int>::iterator itgape=mapBeginBSLS.end(); itgape--;
            countGapLumi_ = ( (itgape->second) - (itgapb->second) );
            //if we see Gap more than then 2*resetNFitLumi !!!!!!!
            //for example if 10-15 is fitted and if 16-25 are missing then we next fit will be for range 11-26 but BS can change in between
            // so better start  as fresh  and reset everything like starting in the begining!
            if(countGapLumi_ >= 2*resetFitNLumi_){RestartFitting(); mapBeginBSLS[countLumi_]   = nthlumi;}
            */
    }

    if (debug_)
      edm::LogInfo("FakeBeamMonitor") << " beginLuminosityBlock::  Size of mapBeginBSLS After = " << mapBeginBSLS.size()
                                      << endl;

    map<int, int>::iterator bbs = mapBeginBSLS.begin();
    map<int, int>::iterator bpv = mapBeginPVLS.begin();
    map<int, std::time_t>::iterator bbst = mapBeginBSTime.begin();
    map<int, std::time_t>::iterator bpvt = mapBeginPVTime.begin();

    if (beginLumiOfPVFit_ == 0)
      beginLumiOfPVFit_ = bpv->second;  //new begin time after removing the LS
    if (beginLumiOfBSFit_ == 0)
      beginLumiOfBSFit_ = bbs->second;
    if (refBStime[0] == 0)
      refBStime[0] = bbst->second;
    if (refPVtime[0] == 0)
      refPVtime[0] = bpvt->second;

  }  //same logic for average fit as above commented line

  map<int, std::time_t>::iterator nbbst = mapBeginBSTime.begin();
  map<int, std::time_t>::iterator nbpvt = mapBeginPVTime.begin();

  if (onlineMode_ && (nthlumi < nextlumi_))
    return;

  if (onlineMode_) {
    if (nthlumi > nextlumi_) {
      if (countLumi_ != 0 && processed_)
        FitAndFill(lumiSeg, lastlumi_, nextlumi_, nthlumi);
      nextlumi_ = nthlumi;
      edm::LogInfo("FakeBeamMonitor") << "beginLuminosityBlock:: Next Lumi to Fit: " << nextlumi_ << endl;
      if ((StartAverage_) && refBStime[0] == 0)
        refBStime[0] = nbbst->second;
      if ((StartAverage_) && refPVtime[0] == 0)
        refPVtime[0] = nbpvt->second;
    }
  } else {
    if (processed_)
      FitAndFill(lumiSeg, lastlumi_, nextlumi_, nthlumi);
    nextlumi_ = nthlumi;
    edm::LogInfo("FakeBeamMonitor") << " beginLuminosityBlock:: Next Lumi to Fit: " << nextlumi_ << endl;
    if ((StartAverage_) && refBStime[0] == 0)
      refBStime[0] = nbbst->second;
    if ((StartAverage_) && refPVtime[0] == 0)
      refPVtime[0] = nbpvt->second;
  }

  //countLumi_++;
  if (processed_)
    processed_ = false;
  edm::LogInfo("FakeBeamMonitor") << " beginLuminosityBlock::  Begin of Lumi: " << nthlumi << endl;
}

// ----------------------------------------------------------
void FakeBeamMonitor::analyze(const Event& iEvent, const EventSetup& iSetup) {
  const int nthlumi = iEvent.luminosityBlock();

  if (onlineMode_ && (nthlumi < nextlumi_)) {
    edm::LogInfo("FakeBeamMonitor") << "analyze::  Spilt event from previous lumi section!" << std::endl;
    return;
  }
  if (onlineMode_ && (nthlumi > nextlumi_)) {
    edm::LogInfo("FakeBeamMonitor") << "analyze::  Spilt event from next lumi section!!!" << std::endl;
    return;
  }

  countEvt_++;
  //  theBeamFitter->readEvent(
  //      iEvent);  //Remember when track fitter read the event in the same place the PVFitter read the events !!!!!!!!!

  //  Handle<reco::BeamSpot> recoBeamSpotHandle;
  //  iEvent.getByToken(bsSrc_, recoBeamSpotHandle);
  //  refBS = *recoBeamSpotHandle;

  //  //------Cut Flow Table filled every event!--------------------------------------
  //  {
  //    // Make a copy of the cut flow table from the beam fitter.
  //    auto tmphisto = static_cast<TH1F*>(theBeamFitter->getCutFlow());
  //    cutFlowTable->getTH1()->SetBins(
  //        tmphisto->GetNbinsX(), tmphisto->GetXaxis()->GetXmin(), tmphisto->GetXaxis()->GetXmax());
  //    // Update the bin labels
  //    if (countEvt_ == 1)  // SetLabel just once
  //      for (int n = 0; n < tmphisto->GetNbinsX(); n++)
  //        cutFlowTable->setBinLabel(n + 1, tmphisto->GetXaxis()->GetBinLabel(n + 1), 1);
  //    cutFlowTable->Reset();
  //    cutFlowTable->getTH1()->Add(tmphisto);
  //  }

  //----Reco tracks -------------------------------------
  //  Handle<reco::TrackCollection> TrackCollection;
  //  iEvent.getByToken(tracksLabel_, TrackCollection);
  //  const reco::TrackCollection* tracks = TrackCollection.product();
  //  for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); ++track) {
  //    h_trkPt->Fill(track->pt());  //no need to change  here for average bs
  //    h_trkVz->Fill(track->vz());
  //  }

  //-------HLT Trigger --------------------------------
  //  edm::Handle<TriggerResults> triggerResults;
  //  bool JetTrigPass = false;
  //  if (iEvent.getByToken(hltSrc_, triggerResults)) {
  //    const edm::TriggerNames& trigNames = iEvent.triggerNames(*triggerResults);
  //    for (unsigned int i = 0; i < triggerResults->size(); i++) {
  //      const std::string& trigName = trigNames.triggerName(i);
  //
  //      if (JetTrigPass)
  //        continue;
  //
  //      for (size_t t = 0; t < jetTrigger_.size(); ++t) {
  //        if (JetTrigPass)
  //          continue;
  //
  //        string string_search(jetTrigger_[t]);
  //        size_t found = trigName.find(string_search);
  //
  //        if (found != string::npos) {
  //          int thisTrigger_ = trigNames.triggerIndex(trigName);
  //          if (triggerResults->accept(thisTrigger_))
  //            JetTrigPass = true;
  //        }  //if trigger found
  //      }    //for(t=0;..)
  //    }      //for(i=0; ..)
  //  }        //if trigger colleciton exist)

  //------ Primary Vertices-------
  //  edm::Handle<reco::VertexCollection> PVCollection;

  //  if (iEvent.getByToken(pvSrc_, PVCollection)) {
  int nPVcount = 0;
  int nPVcount_ST = 0;  //For Single Trigger(hence ST)

  //    for (reco::VertexCollection::const_iterator pv = PVCollection->begin(); pv != PVCollection->end(); ++pv) {
  for (int tmp_idx = 0; tmp_idx < 10; tmp_idx++) {
    //--- vertex selection
    //    if (pv->isFake() || pv->tracksSize() == 0)
    //      continue;
    nPVcount++;  // count non fake pv:

    //if (JetTrigPass)
    nPVcount_ST++;  //non-fake pv with a specific trigger

    //    if (pv->ndof() < minVtxNdf_ || (pv->ndof() + 3.) / pv->tracksSize() < 2 * minVtxWgt_)
    //      continue;

    //Fill this map to store xyx for pv so that later we can remove the first one for run aver
    mapPVx[countLumi_].push_back(tmp_idx);
    mapPVy[countLumi_].push_back(tmp_idx);
    mapPVz[countLumi_].push_back(tmp_idx);

    //      if (!StartAverage_) {  //for first N LS
    //        h_PVx[0]->Fill(pv->x());
    //        h_PVy[0]->Fill(pv->y());
    //        h_PVz[0]->Fill(pv->z());
    //        h_PVxz->Fill(pv->z(), pv->x());
    //        h_PVyz->Fill(pv->z(), pv->y());
    //      }  //for first N LiS
    //      else {
    //        h_PVxz->Fill(pv->z(), pv->x());
    //        h_PVyz->Fill(pv->z(), pv->y());
    //      }

  }  //loop over pvs

  //    h_nVtx->Fill(nPVcount * 1.);  //no need to change it for average BS

  mapNPV[countLumi_].push_back((nPVcount_ST));

  //    if (!StartAverage_) {
  //      h_nVtx_st->Fill(nPVcount_ST * 1.);
  //    }

  //  }  //if pv collection is availaable

  if (StartAverage_) {
    map<int, std::vector<float> >::iterator itpvx = mapPVx.begin();
    map<int, std::vector<float> >::iterator itpvy = mapPVy.begin();
    map<int, std::vector<float> >::iterator itpvz = mapPVz.begin();

    map<int, std::vector<int> >::iterator itbspvinfo = mapNPV.begin();

    if ((int)mapPVx.size() > resetFitNLumi_) {  //sometimes the events is not there but LS is there!
      mapPVx.erase(itpvx);
      mapPVy.erase(itpvy);
      mapPVz.erase(itpvz);
      mapNPV.erase(itbspvinfo);
    }  //loop over Last N lumi collected

  }  //StartAverage==true

  processed_ = true;
}

//--------------------------------------------------------
void FakeBeamMonitor::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& iSetup) {
  int nthlumi = lumiSeg.id().luminosityBlock();
  edm::LogInfo("FakeBeamMonitor") << "endLuminosityBlock:: Lumi of the last event before endLuminosityBlock: "
                                  << nthlumi << endl;

  if (onlineMode_ && nthlumi < nextlumi_)
    return;
  const edm::TimeValue_t fendtimestamp = lumiSeg.endTime().value();
  const std::time_t fendtime = fendtimestamp >> 32;
  tmpTime = refBStime[1] = refPVtime[1] = fendtime;

  // end DB logger
  if (onlineDbService_.isAvailable()) {
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor::endLuminosityBlock";
    onlineDbService_->logger().end(DBloggerReturn_);
  }
}

//--------------------------------------------------------
void FakeBeamMonitor::FitAndFill(const LuminosityBlock& lumiSeg, int& lastlumi, int& nextlumi, int& nthlumi) {
  if (onlineMode_ && (nthlumi <= nextlumi))
    return;

  //set the correct run number when no event in the LS for fake output
  //  if ((processed_) && theBeamFitter->getRunNumber() != frun)
  //    theBeamFitter->setRun(frun);

  int currentlumi = nextlumi;
  edm::LogInfo("FakeBeamMonitor") << "FitAndFill::  Lumi of the current fit: " << currentlumi << endl;
  lastlumi = currentlumi;
  endLumiOfBSFit_ = currentlumi;
  endLumiOfPVFit_ = currentlumi;

  //---------Fix for Runninv average-------------
  mapLSPVStoreSize[countLumi_] = 10;  //theBeamFitter->getPVvectorSize();

  //  if (StartAverage_) {
  //    std::map<int, std::size_t>::iterator rmLSPVi = mapLSPVStoreSize.begin();
  //    size_t SizeToRemovePV = rmLSPVi->second;
  //    for (std::map<int, std::size_t>::iterator rmLSPVe = mapLSPVStoreSize.end(); ++rmLSPVi != rmLSPVe;)
  //      rmLSPVi->second -= SizeToRemovePV;
  //
  //    theBeamFitter->resizePVvector(SizeToRemovePV);
  //
  //    map<int, std::size_t>::iterator tmpItpv = mapLSPVStoreSize.begin();
  //    mapLSPVStoreSize.erase(tmpItpv);
  //  }
  //  if (debug_)
  //    edm::LogInfo("BeamMonitor") << "FitAndFill:: Size of thePVvector After removing the PVs = "
  //                                << theBeamFitter->getPVvectorSize() << endl;

  //lets filt the PV for GUI here: It was in analyzer in preivous versiton but moved here due to absence of event in some lumis, works OK
  bool resetHistoFlag_ = false;
  if ((int)mapPVx.size() >= resetFitNLumi_ && (StartAverage_)) {
    h_PVx[0]->Reset();
    h_PVy[0]->Reset();
    h_PVz[0]->Reset();
    h_nVtx_st->Reset();
    resetHistoFlag_ = true;
  }

  int MaxPVs = 0;
  int countEvtLastNLS_ = 0;
  int countTotPV_ = 0;

  std::map<int, std::vector<int> >::iterator mnpv = mapNPV.begin();
  std::map<int, std::vector<float> >::iterator mpv2 = mapPVy.begin();
  std::map<int, std::vector<float> >::iterator mpv3 = mapPVz.begin();

  for (std::map<int, std::vector<float> >::iterator mpv1 = mapPVx.begin(); mpv1 != mapPVx.end();
       ++mpv1, ++mpv2, ++mpv3, ++mnpv) {
    std::vector<float>::iterator mpvs2 = (mpv2->second).begin();
    std::vector<float>::iterator mpvs3 = (mpv3->second).begin();
    for (std::vector<float>::iterator mpvs1 = (mpv1->second).begin(); mpvs1 != (mpv1->second).end();
         ++mpvs1, ++mpvs2, ++mpvs3) {
      if (resetHistoFlag_) {
        h_PVx[0]->Fill(*mpvs1);  //these histogram are reset after StartAverage_ flag is ON
        h_PVy[0]->Fill(*mpvs2);
        h_PVz[0]->Fill(*mpvs3);
      }
    }  //loop over second

    //Do the same here for nPV distr.
    for (std::vector<int>::iterator mnpvs = (mnpv->second).begin(); mnpvs != (mnpv->second).end(); ++mnpvs) {
      if ((*mnpvs > 0) && (resetHistoFlag_))
        h_nVtx_st->Fill((*mnpvs) * (1.0));
      countEvtLastNLS_++;
      countTotPV_ += (*mnpvs);
      if ((*mnpvs) > MaxPVs)
        MaxPVs = (*mnpvs);
    }  //loop over second of mapNPV

  }  //loop over last N lumis

  char tmpTitlePV[100];
  sprintf(tmpTitlePV, "%s %i %s %i", "Num. of reco. vertices for LS: ", beginLumiOfPVFit_, " to ", endLumiOfPVFit_);
  h_nVtx_st->setAxisTitle(tmpTitlePV, 1);

  //  std::vector<float> DipPVInfo_;
  //  DipPVInfo_.clear();
  //
  //  if (countTotPV_ != 0) {
  //    DipPVInfo_.push_back((float)countEvtLastNLS_);
  //    DipPVInfo_.push_back(h_nVtx_st->getMean());
  //    DipPVInfo_.push_back(h_nVtx_st->getMeanError());
  //    DipPVInfo_.push_back(h_nVtx_st->getRMS());
  //    DipPVInfo_.push_back(h_nVtx_st->getRMSError());
  //    DipPVInfo_.push_back((float)MaxPVs);
  //    DipPVInfo_.push_back((float)countTotPV_);
  //    MaxPVs = 0;
  //  } else {
  //    for (size_t i = 0; i < 7; i++) {
  //      if (i > 0) {
  //        DipPVInfo_.push_back(0.);
  //      } else {
  //        DipPVInfo_.push_back((float)countEvtLastNLS_);
  //      }
  //    }
  //  }
  //  theBeamFitter->SetPVInfo(DipPVInfo_);
  countEvtLastNLS_ = 0;

  if (onlineMode_) {  // filling LS gap
    // FIXME: need to add protection for the case if the gap is at the resetting LS!
    const int countLS_bs = hs[k_x0_lumi]->getTH1()->GetEntries();
    const int countLS_pv = hs[k_PVx_lumi]->getTH1()->GetEntries();
    edm::LogInfo("FakeBeamMonitor") << "FitAndFill:: countLS_bs = " << countLS_bs << " ; countLS_pv = " << countLS_pv
                                    << std::endl;
    int LSgap_bs = currentlumi / fitNLumi_ - countLS_bs;
    int LSgap_pv = currentlumi / fitPVNLumi_ - countLS_pv;
    if (currentlumi % fitNLumi_ == 0)
      LSgap_bs--;
    if (currentlumi % fitPVNLumi_ == 0)
      LSgap_pv--;
    edm::LogInfo("FakeBeamMonitor") << "FitAndFill::  LSgap_bs = " << LSgap_bs << " ; LSgap_pv = " << LSgap_pv
                                    << std::endl;
    // filling previous fits if LS gap ever exists
    for (int ig = 0; ig < LSgap_bs; ig++) {
      hs[k_x0_lumi]->ShiftFillLast(0., 0., fitNLumi_);  //x0 , x0err, fitNLumi_;  see DQMCore....
      hs[k_y0_lumi]->ShiftFillLast(0., 0., fitNLumi_);
      hs[k_z0_lumi]->ShiftFillLast(0., 0., fitNLumi_);
      hs[k_sigmaX0_lumi]->ShiftFillLast(0., 0., fitNLumi_);
      hs[k_sigmaY0_lumi]->ShiftFillLast(0., 0., fitNLumi_);
      hs[k_sigmaZ0_lumi]->ShiftFillLast(0., 0., fitNLumi_);
      h_nVtx_lumi->ShiftFillLast(0., 0., fitNLumi_);
    }
    for (int ig = 0; ig < LSgap_pv; ig++) {
      hs[k_PVx_lumi]->ShiftFillLast(0., 0., fitPVNLumi_);
      hs[k_PVy_lumi]->ShiftFillLast(0., 0., fitPVNLumi_);
      hs[k_PVz_lumi]->ShiftFillLast(0., 0., fitPVNLumi_);
    }
    const int previousLS = h_nTrk_lumi->getTH1()->GetEntries();
    for (int i = 1; i < (currentlumi - previousLS);
         i++)  //if (current-previoius)= 1 then never go inside the for loop!!!!!!!!!!!
      h_nTrk_lumi->ShiftFillLast(nthBSTrk_);
  }

  edm::LogInfo("FakeBeamMonitor") << "FitAndFill:: Time lapsed since last scroll = " << tmpTime - refTime << std::endl;

  if (testScroll(tmpTime, refTime)) {
    scrollTH1(hs[k_x0_time]->getTH1(), refTime);
    scrollTH1(hs[k_y0_time]->getTH1(), refTime);
    scrollTH1(hs[k_z0_time]->getTH1(), refTime);
    scrollTH1(hs[k_sigmaX0_time]->getTH1(), refTime);
    scrollTH1(hs[k_sigmaY0_time]->getTH1(), refTime);
    scrollTH1(hs[k_sigmaZ0_time]->getTH1(), refTime);
    scrollTH1(hs[k_PVx_time]->getTH1(), refTime);
    scrollTH1(hs[k_PVy_time]->getTH1(), refTime);
    scrollTH1(hs[k_PVz_time]->getTH1(), refTime);
  }

  //  bool doPVFit = false;
  //
  //  if (fitPVNLumi_ > 0) {
  //    if (onlineMode_) {
  //      if (currentlumi % fitPVNLumi_ == 0)
  //        doPVFit = true;
  //    } else if (countLumi_ % fitPVNLumi_ == 0)
  //      doPVFit = true;
  //  } else
  //    doPVFit = true;
  //
  //  if (doPVFit) {
  edm::LogInfo("FakeBeamMonitor") << "FitAndFill:: Do PV Fitting for LS = " << beginLumiOfPVFit_ << " to "
                                  << endLumiOfPVFit_ << std::endl;
  // Primary Vertex Fit:
  if (h_PVx[0]->getTH1()->GetEntries() > minNrVertices_) {
    pvResults->Reset();
    char tmpTitle[50];
    sprintf(tmpTitle, "%s %i %s %i", "Fitted Primary Vertex (cm) of LS: ", beginLumiOfPVFit_, " to ", endLumiOfPVFit_);
    pvResults->setAxisTitle(tmpTitle, 1);

    std::unique_ptr<TF1> fgaus{new TF1("fgaus", "gaus")};
    double mean, width, meanErr, widthErr;
    fgaus->SetLineColor(4);
    h_PVx[0]->getTH1()->Fit(fgaus.get(), "QLM0");
    mean = fgaus->GetParameter(1);
    width = fgaus->GetParameter(2);
    meanErr = fgaus->GetParError(1);
    widthErr = fgaus->GetParError(2);

    hs[k_PVx_lumi]->ShiftFillLast(mean, width, fitPVNLumi_);
    hs[k_PVx_lumi_all]->setBinContent(currentlumi, mean);
    hs[k_PVx_lumi_all]->setBinError(currentlumi, width);
    int nthBin = tmpTime - refTime;
    if (nthBin < 0)
      edm::LogInfo("FakeBeamMonitor") << "FitAndFill::  Event time outside current range of time histograms!"
                                      << std::endl;
    if (nthBin > 0) {
      hs[k_PVx_time]->setBinContent(nthBin, mean);
      hs[k_PVx_time]->setBinError(nthBin, width);
    }
    int jthBin = tmpTime - startTime;
    if (jthBin > 0) {
      hs[k_PVx_time_all]->setBinContent(jthBin, mean);
      hs[k_PVx_time_all]->setBinError(jthBin, width);
    }
    pvResults->setBinContent(1, 6, mean);
    pvResults->setBinContent(1, 3, width);
    pvResults->setBinContent(2, 6, meanErr);
    pvResults->setBinContent(2, 3, widthErr);

    {
      // snap shot of the fit
      auto tmphisto = h_PVx[0]->getTH1F();
      h_PVx[1]->getTH1()->SetBins(
          tmphisto->GetNbinsX(), tmphisto->GetXaxis()->GetXmin(), tmphisto->GetXaxis()->GetXmax());
      h_PVx[1]->Reset();
      h_PVx[1]->getTH1()->Add(tmphisto);
      h_PVx[1]->getTH1()->Fit(fgaus.get(), "QLM");
    }

    h_PVy[0]->getTH1()->Fit(fgaus.get(), "QLM0");
    mean = fgaus->GetParameter(1);
    width = fgaus->GetParameter(2);
    meanErr = fgaus->GetParError(1);
    widthErr = fgaus->GetParError(2);
    hs[k_PVy_lumi]->ShiftFillLast(mean, width, fitPVNLumi_);
    hs[k_PVy_lumi_all]->setBinContent(currentlumi, mean);
    hs[k_PVy_lumi_all]->setBinError(currentlumi, width);
    if (nthBin > 0) {
      hs[k_PVy_time]->setBinContent(nthBin, mean);
      hs[k_PVy_time]->setBinError(nthBin, width);
    }
    if (jthBin > 0) {
      hs[k_PVy_time_all]->setBinContent(jthBin, mean);
      hs[k_PVy_time_all]->setBinError(jthBin, width);
    }
    pvResults->setBinContent(1, 5, mean);
    pvResults->setBinContent(1, 2, width);
    pvResults->setBinContent(2, 5, meanErr);
    pvResults->setBinContent(2, 2, widthErr);
    // snap shot of the fit
    {
      auto tmphisto = h_PVy[0]->getTH1F();
      h_PVy[1]->getTH1()->SetBins(
          tmphisto->GetNbinsX(), tmphisto->GetXaxis()->GetXmin(), tmphisto->GetXaxis()->GetXmax());
      h_PVy[1]->Reset();
      h_PVy[1]->getTH1()->Add(tmphisto);
      h_PVy[1]->getTH1()->Fit(fgaus.get(), "QLM");
    }

    h_PVz[0]->getTH1()->Fit(fgaus.get(), "QLM0");
    mean = fgaus->GetParameter(1);
    width = fgaus->GetParameter(2);
    meanErr = fgaus->GetParError(1);
    widthErr = fgaus->GetParError(2);
    hs[k_PVz_lumi]->ShiftFillLast(mean, width, fitPVNLumi_);
    hs[k_PVz_lumi_all]->setBinContent(currentlumi, mean);
    hs[k_PVz_lumi_all]->setBinError(currentlumi, width);
    if (nthBin > 0) {
      hs[k_PVz_time]->setBinContent(nthBin, mean);
      hs[k_PVz_time]->setBinError(nthBin, width);
    }
    if (jthBin > 0) {
      hs[k_PVz_time_all]->setBinContent(jthBin, mean);
      hs[k_PVz_time_all]->setBinError(jthBin, width);
    }
    pvResults->setBinContent(1, 4, mean);
    pvResults->setBinContent(1, 1, width);
    pvResults->setBinContent(2, 4, meanErr);
    pvResults->setBinContent(2, 1, widthErr);
    {
      // snap shot of the fit
      auto tmphisto = h_PVz[0]->getTH1F();
      h_PVz[1]->getTH1()->SetBins(
          tmphisto->GetNbinsX(), tmphisto->GetXaxis()->GetXmin(), tmphisto->GetXaxis()->GetXmax());
      h_PVz[1]->Reset();
      h_PVz[1]->getTH1()->Add(tmphisto);
      h_PVz[1]->getTH1()->Fit(fgaus.get(), "QLM");
    }
  }  //check if found min Vertices
     //  }    //do PVfit

  if ((resetPVNLumi_ > 0 && countLumi_ == resetPVNLumi_) || StartAverage_) {
    beginLumiOfPVFit_ = 0;
    refPVtime[0] = 0;
  }

  //---------Readjustment of theBSvector, RefTime, beginLSofFit---------
  //  vector<BSTrkParameters> theBSvector1 = theBeamFitter->getBSvector();
  //  mapLSBSTrkSize[countLumi_] = (theBSvector1.size());
  size_t PreviousRecords = 0;  //needed to fill nth record of tracks in GUI

  //  if (StartAverage_) {
  //    size_t SizeToRemove = 0;
  //    std::map<int, std::size_t>::iterator rmls = mapLSBSTrkSize.begin();
  //    SizeToRemove = rmls->second;
  //    if (debug_)
  //      edm::LogInfo("BeamMonitor") << "  The size to remove is =  " << SizeToRemove << endl;
  //    int changedAfterThis = 0;
  //    for (std::map<int, std::size_t>::iterator rmLS = mapLSBSTrkSize.begin(); rmLS != mapLSBSTrkSize.end();
  //         ++rmLS, ++changedAfterThis) {
  //      if (changedAfterThis > 0) {
  //        (rmLS->second) = (rmLS->second) - SizeToRemove;
  //        if ((mapLSBSTrkSize.size() - (size_t)changedAfterThis) == 2)
  //          PreviousRecords = (rmLS->second);
  //      }
  //    }
  //
  //    theBeamFitter->resizeBSvector(SizeToRemove);
  //
  //    map<int, std::size_t>::iterator tmpIt = mapLSBSTrkSize.begin();
  //    mapLSBSTrkSize.erase(tmpIt);
  //
  //    std::pair<int, int> checkfitLS = theBeamFitter->getFitLSRange();
  //    std::pair<time_t, time_t> checkfitTime = theBeamFitter->getRefTime();
  //    theBeamFitter->setFitLSRange(beginLumiOfBSFit_, checkfitLS.second);
  //    theBeamFitter->setRefTime(refBStime[0], checkfitTime.second);
  //  }

  //Fill the track for this fit
  //  vector<BSTrkParameters> theBSvector = theBeamFitter->getBSvector();
  //  h_nTrk_lumi->ShiftFillLast(theBSvector.size());
  //
  //  if (debug_)
  //    edm::LogInfo("BeamMonitor") << "FitAndFill::   Size of  theBSViector.size()  After =" << theBSvector.size() << endl;

  //  bool countFitting = false;
  //  if (theBSvector.size() >= PreviousRecords && theBSvector.size() >= min_Ntrks_) {
  //    countFitting = true;
  //  }

  //---Fix for Cut Flow Table for Running average in a same way//the previous code  has problem for resetting!!!
  //  mapLSCF[countLumi_] = *theBeamFitter->getCutFlow();
  //  if (StartAverage_ && !mapLSCF.empty()) {
  //    const TH1F& cutFlowToSubtract = mapLSCF.begin()->second;
  //    // Subtract the last cut flow from all of the others.
  //    std::map<int, TH1F>::iterator cf = mapLSCF.begin();
  //    // Start on second entry
  //    for (; cf != mapLSCF.end(); ++cf) {
  //      cf->second.Add(&cutFlowToSubtract, -1);
  //    }
  //    theBeamFitter->subtractFromCutFlow(&cutFlowToSubtract);
  //    // Remove the obsolete lumi section
  //    mapLSCF.erase(mapLSCF.begin());
  //  }

  if (resetHistos_) {
    h_d0_phi0->Reset();
    h_vx_vy->Reset();
    h_vx_dz->Reset();
    h_vy_dz->Reset();
    h_trk_z0->Reset();
    resetHistos_ = false;
  }

  if (StartAverage_)
    nthBSTrk_ = PreviousRecords;  //after average proccess is ON//for 2-6 LS fit PreviousRecords is size from 2-5 LS

  edm::LogInfo("FakeBeamMonitor") << " The Previous Recored for this fit is  =" << nthBSTrk_ << endl;

  //  unsigned int itrk = 0;
  //  for (vector<BSTrkParameters>::const_iterator BSTrk = theBSvector.begin(); BSTrk != theBSvector.end();
  //       ++BSTrk, ++itrk) {
  //    if (itrk >= nthBSTrk_) {  //fill for this record only !!
  //      h_d0_phi0->Fill(BSTrk->phi0(), BSTrk->d0());
  //      double vx = BSTrk->vx();
  //      double vy = BSTrk->vy();
  //      double z0 = BSTrk->z0();
  //      h_vx_vy->Fill(vx, vy);
  //      h_vx_dz->Fill(z0, vx);
  //      h_vy_dz->Fill(z0, vy);
  //      h_trk_z0->Fill(z0);
  //    }
  //  }

  //  nthBSTrk_ = theBSvector.size();  // keep track of num of tracks filled so far

  edm::LogInfo("FakeBeamMonitor") << " The Current Recored for this fit is  =" << nthBSTrk_ << endl;

  //  if (countFitting)
  //    edm::LogInfo("FakeBeamMonitor") << "FitAndFill::  Num of tracks collected = " << nthBSTrk_ << endl;

  if (fitNLumi_ > 0) {
    if (onlineMode_) {
      if (currentlumi % fitNLumi_ != 0) {
        // 	for (std::map<TString,MonitorElement*>::iterator itAll = hs.begin();
        // 	     itAll != hs.end(); ++itAll) {
        // 	  if ((*itAll).first.Contains("all")) {
        // 	    (*itAll).second->setBinContent(currentlumi,0.);
        // 	    (*itAll).second->setBinError(currentlumi,0.);
        // 	  }
        // 	}
        return;
      }
    } else if (countLumi_ % fitNLumi_ != 0)
      return;
  }

  edm::LogInfo("FakeBeamMonitor") << "FitAndFill:: [DebugTime] refBStime[0] = " << refBStime[0]
                                  << "; address =  " << &refBStime[0] << std::endl;
  edm::LogInfo("FakeBeamMonitor") << "FitAndFill:: [DebugTime] refBStime[1] = " << refBStime[1]
                                  << "; address =  " << &refBStime[1] << std::endl;

  //Fill for all LS even if fit fails
  //  h_nVtx_lumi->ShiftFillLast((theBeamFitter->getPVvectorSize()), 0., fitNLumi_);
  //  h_nVtx_lumi_all->setBinContent(currentlumi, (theBeamFitter->getPVvectorSize()));

  //  if (countFitting) {
  nFits_++;
  //    std::pair<int, int> fitLS = theBeamFitter->getFitLSRange();
  std::pair<int, int> fitLS(beginLumiOfBSFit_, endLumiOfBSFit_);
  //    edm::LogInfo("BeamMonitor") << "FitAndFill::  [BeamFitter] Do BeamSpot Fit for LS = " << fitLS.first << " to "
  //                                << fitLS.second << std::endl;
  edm::LogInfo("FakeBeamMonitor") << "FitAndFill::  [FakeBeamMonitor] Do BeamSpot Fit for LS = " << beginLumiOfBSFit_
                                  << " to " << endLumiOfBSFit_ << std::endl;

  //Now Run the PV and Track Fitter over the collected tracks and pvs
  //    if (theBeamFitter->runPVandTrkFitter()) {
  //      reco::BeamSpot bs = theBeamFitter->getBeamSpot();

  //Create fake BS here
  reco::BeamSpot::CovarianceMatrix matrix;
  for (int j = 0; j < 7; ++j) {
    for (int k = j; k < 7; ++k) {
      matrix(j, k) = 0;
    }
  }

  // random values for fake BeamSpot
  float tmp_BSx = rndm_->Gaus(0.1, 0.1);            // [cm]
  float tmp_BSy = rndm_->Gaus(0.1, 0.1);            // [cm]
  float tmp_BSz = rndm_->Gaus(0.1, 0.1);            // [cm]
  float tmp_BSwidthX = rndm_->Gaus(0.001, 0.0005);  // [cm]
  float tmp_BSwidthY = rndm_->Gaus(0.001, 0.0005);  // [cm]
  float tmp_BSwidthZ = rndm_->Gaus(3.5, 0.5);       // [cm]

  reco::BeamSpot bs(reco::BeamSpot::Point(tmp_BSx, tmp_BSy, tmp_BSz),
                    tmp_BSwidthZ,
                    0,
                    0,
                    tmp_BSwidthX,
                    matrix,
                    reco::BeamSpot::Tracker);
  bs.setBeamWidthY(tmp_BSwidthY);

  if (bs.type() > 0)  // with good beamwidth fit
    preBS = bs;       // cache good fit results

  edm::LogInfo("FakeBeamMonitor") << "\n RESULTS OF DEFAULT FIT:" << endl;
  edm::LogInfo("FakeBeamMonitor") << bs << endl;
  edm::LogInfo("FakeBeamMonitor") << "[BeamFitter] fitting done \n" << endl;

  hs[k_x0_lumi]->ShiftFillLast(bs.x0(), bs.x0Error(), fitNLumi_);
  hs[k_y0_lumi]->ShiftFillLast(bs.y0(), bs.y0Error(), fitNLumi_);
  hs[k_z0_lumi]->ShiftFillLast(bs.z0(), bs.z0Error(), fitNLumi_);
  hs[k_sigmaX0_lumi]->ShiftFillLast(bs.BeamWidthX(), bs.BeamWidthXError(), fitNLumi_);
  hs[k_sigmaY0_lumi]->ShiftFillLast(bs.BeamWidthY(), bs.BeamWidthYError(), fitNLumi_);
  hs[k_sigmaZ0_lumi]->ShiftFillLast(bs.sigmaZ(), bs.sigmaZ0Error(), fitNLumi_);
  hs[k_x0_lumi_all]->setBinContent(currentlumi, bs.x0());
  hs[k_x0_lumi_all]->setBinError(currentlumi, bs.x0Error());
  hs[k_y0_lumi_all]->setBinContent(currentlumi, bs.y0());
  hs[k_y0_lumi_all]->setBinError(currentlumi, bs.y0Error());
  hs[k_z0_lumi_all]->setBinContent(currentlumi, bs.z0());
  hs[k_z0_lumi_all]->setBinError(currentlumi, bs.z0Error());
  hs[k_sigmaX0_lumi_all]->setBinContent(currentlumi, bs.BeamWidthX());
  hs[k_sigmaX0_lumi_all]->setBinError(currentlumi, bs.BeamWidthXError());
  hs[k_sigmaY0_lumi_all]->setBinContent(currentlumi, bs.BeamWidthY());
  hs[k_sigmaY0_lumi_all]->setBinError(currentlumi, bs.BeamWidthYError());
  hs[k_sigmaZ0_lumi_all]->setBinContent(currentlumi, bs.sigmaZ());
  hs[k_sigmaZ0_lumi_all]->setBinError(currentlumi, bs.sigmaZ0Error());

  int nthBin = tmpTime - refTime;
  if (nthBin > 0) {
    hs[k_x0_time]->setBinContent(nthBin, bs.x0());
    hs[k_y0_time]->setBinContent(nthBin, bs.y0());
    hs[k_z0_time]->setBinContent(nthBin, bs.z0());
    hs[k_sigmaX0_time]->setBinContent(nthBin, bs.BeamWidthX());
    hs[k_sigmaY0_time]->setBinContent(nthBin, bs.BeamWidthY());
    hs[k_sigmaZ0_time]->setBinContent(nthBin, bs.sigmaZ());
    hs[k_x0_time]->setBinError(nthBin, bs.x0Error());
    hs[k_y0_time]->setBinError(nthBin, bs.y0Error());
    hs[k_z0_time]->setBinError(nthBin, bs.z0Error());
    hs[k_sigmaX0_time]->setBinError(nthBin, bs.BeamWidthXError());
    hs[k_sigmaY0_time]->setBinError(nthBin, bs.BeamWidthYError());
    hs[k_sigmaZ0_time]->setBinError(nthBin, bs.sigmaZ0Error());
  }

  int jthBin = tmpTime - startTime;
  if (jthBin > 0) {
    hs[k_x0_time_all]->setBinContent(jthBin, bs.x0());
    hs[k_y0_time_all]->setBinContent(jthBin, bs.y0());
    hs[k_z0_time_all]->setBinContent(jthBin, bs.z0());
    hs[k_sigmaX0_time_all]->setBinContent(jthBin, bs.BeamWidthX());
    hs[k_sigmaY0_time_all]->setBinContent(jthBin, bs.BeamWidthY());
    hs[k_sigmaZ0_time_all]->setBinContent(jthBin, bs.sigmaZ());
    hs[k_x0_time_all]->setBinError(jthBin, bs.x0Error());
    hs[k_y0_time_all]->setBinError(jthBin, bs.y0Error());
    hs[k_z0_time_all]->setBinError(jthBin, bs.z0Error());
    hs[k_sigmaX0_time_all]->setBinError(jthBin, bs.BeamWidthXError());
    hs[k_sigmaY0_time_all]->setBinError(jthBin, bs.BeamWidthYError());
    hs[k_sigmaZ0_time_all]->setBinError(jthBin, bs.sigmaZ0Error());
  }

  h_x0->Fill(bs.x0());
  h_y0->Fill(bs.y0());
  h_z0->Fill(bs.z0());
  if (bs.type() > 0) {  // with good beamwidth fit
    h_sigmaX0->Fill(bs.BeamWidthX());
    h_sigmaY0->Fill(bs.BeamWidthY());
  }
  h_sigmaZ0->Fill(bs.sigmaZ());

  if (nthBSTrk_ >= 2 * min_Ntrks_) {
    double amp = std::sqrt(bs.x0() * bs.x0() + bs.y0() * bs.y0());
    double alpha = std::atan2(bs.y0(), bs.x0());
    std::unique_ptr<TF1> f1{new TF1("f1", "[0]*sin(x-[1])", -3.14, 3.14)};
    f1->SetParameters(amp, alpha);
    f1->SetParLimits(0, amp - 0.1, amp + 0.1);
    f1->SetParLimits(1, alpha - 0.577, alpha + 0.577);
    f1->SetLineColor(4);
    h_d0_phi0->getTProfile()->Fit(f1.get(), "QR");

    double mean = bs.z0();
    double width = bs.sigmaZ();
    std::unique_ptr<TF1> fgaus{new TF1("fgaus", "gaus")};
    fgaus->SetParameters(mean, width);
    fgaus->SetLineColor(4);
    h_trk_z0->getTH1()->Fit(fgaus.get(), "QLRM", "", mean - 3 * width, mean + 3 * width);
  }

  fitResults->Reset();
  std::pair<int, int> LSRange(beginLumiOfBSFit_, endLumiOfBSFit_);  //= theBeamFitter->getFitLSRange();
  char tmpTitle[50];
  sprintf(tmpTitle, "%s %i %s %i", "Fitted Beam Spot (cm) of LS: ", LSRange.first, " to ", LSRange.second);
  fitResults->setAxisTitle(tmpTitle, 1);
  fitResults->setBinContent(1, 8, bs.x0());
  fitResults->setBinContent(1, 7, bs.y0());
  fitResults->setBinContent(1, 6, bs.z0());
  fitResults->setBinContent(1, 5, bs.sigmaZ());
  fitResults->setBinContent(1, 4, bs.dxdz());
  fitResults->setBinContent(1, 3, bs.dydz());
  if (bs.type() > 0) {  // with good beamwidth fit
    fitResults->setBinContent(1, 2, bs.BeamWidthX());
    fitResults->setBinContent(1, 1, bs.BeamWidthY());
  } else {  // fill cached widths
    fitResults->setBinContent(1, 2, preBS.BeamWidthX());
    fitResults->setBinContent(1, 1, preBS.BeamWidthY());
  }

  fitResults->setBinContent(2, 8, bs.x0Error());
  fitResults->setBinContent(2, 7, bs.y0Error());
  fitResults->setBinContent(2, 6, bs.z0Error());
  fitResults->setBinContent(2, 5, bs.sigmaZ0Error());
  fitResults->setBinContent(2, 4, bs.dxdzError());
  fitResults->setBinContent(2, 3, bs.dydzError());
  if (bs.type() > 0) {  // with good beamwidth fit
    fitResults->setBinContent(2, 2, bs.BeamWidthXError());
    fitResults->setBinContent(2, 1, bs.BeamWidthYError());
  } else {  // fill cached width errors
    fitResults->setBinContent(2, 2, preBS.BeamWidthXError());
    fitResults->setBinContent(2, 1, preBS.BeamWidthYError());
  }

  // count good fit
  //     if (std::fabs(refBS.x0()-bs.x0())/bs.x0Error() < deltaSigCut_) { // disabled temporarily
  summaryContent_[0] += 1.;
  //     }
  //     if (std::fabs(refBS.y0()-bs.y0())/bs.y0Error() < deltaSigCut_) { // disabled temporarily
  summaryContent_[1] += 1.;
  //     }
  //     if (std::fabs(refBS.z0()-bs.z0())/bs.z0Error() < deltaSigCut_) { // disabled temporarily
  summaryContent_[2] += 1.;
  //     }

  // Create the BeamSpotOnlineObjects object
  BeamSpotOnlineObjects* BSOnline = new BeamSpotOnlineObjects();
  BSOnline->SetLastAnalyzedLumi(fitLS.second);
  BSOnline->SetLastAnalyzedRun(frun);
  BSOnline->SetLastAnalyzedFill(0);  // To be updated with correct LHC Fill number
  BSOnline->SetPosition(bs.x0(), bs.y0(), bs.z0());
  BSOnline->SetSigmaZ(bs.sigmaZ());
  BSOnline->SetBeamWidthX(bs.BeamWidthX());
  BSOnline->SetBeamWidthY(bs.BeamWidthY());
  BSOnline->SetBeamWidthXError(bs.BeamWidthXError());
  BSOnline->SetBeamWidthYError(bs.BeamWidthYError());
  BSOnline->Setdxdz(bs.dxdz());
  BSOnline->Setdydz(bs.dydz());
  BSOnline->SetType(bs.type());
  BSOnline->SetEmittanceX(bs.emittanceX());
  BSOnline->SetEmittanceY(bs.emittanceY());
  BSOnline->SetBetaStar(bs.betaStar());
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      BSOnline->SetCovariance(i, j, bs.covariance(i, j));
    }
  }
  //      BSOnline->SetNumTracks(theBeamFitter->getNTracks());
  //      BSOnline->SetNumPVs(theBeamFitter->getNPVs());
  BSOnline->SetNumTracks(50);
  BSOnline->SetNumPVs(10);
  auto creationTime =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  BSOnline->SetCreationTime(creationTime);

  edm::LogInfo("FakeBeamMonitor") << "FitAndFill::[PayloadCreation] BeamSpotOnline object created: \n" << std::endl;
  edm::LogInfo("FakeBeamMonitor") << *BSOnline << std::endl;
  //std::cout << "------------------> fitted BS: " << *BSOnline << std::endl;

  // Create the payload for BeamSpotOnlineObjects object
  if (onlineDbService_.isAvailable()) {
    edm::LogInfo("FakeBeamMonitor") << "FitAndFill::[PayloadCreation] onlineDbService available \n" << std::endl;
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor::FitAndFill - Lumi of the current fit: " << currentlumi;
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor::FitAndFill - Do PV Fitting for LS = " << beginLumiOfPVFit_
                                         << " to " << endLumiOfPVFit_;
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor::FitAndFill - [BeamFitter] Do BeamSpot Fit for LS = "
                                         << fitLS.first << " to " << fitLS.second;
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor::FitAndFill - [FakeBeamMonitor] Do BeamSpot Fit for LS = "
                                         << beginLumiOfBSFit_ << " to " << endLumiOfBSFit_;
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor - RESULTS OF DEFAULT FIT:";
    onlineDbService_->logger().logInfo() << "\n" << bs;
    onlineDbService_->logger().logInfo()
        << "FakeBeamMonitor::FitAndFill - [PayloadCreation] BeamSpotOnline object created:";
    onlineDbService_->logger().logInfo() << "\n" << *BSOnline;
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor::FitAndFill - [PayloadCreation] onlineDbService available";
    onlineDbService_->logger().logInfo() << "FakeBeamMonitor::FitAndFill - [PayloadCreation] SetCreationTime: "
                                         << creationTime << " [epoch in microseconds]";
    try {
      onlineDbService_->writeForNextLumisection(BSOnline, recordName_);
    } catch (const std::exception& e) {
      onlineDbService_->logger().logError() << "FakeBeamMonitor - Error writing record: " << recordName_
                                            << " for Run: " << frun << " - Lumi: " << fitLS.second;
      onlineDbService_->logger().logError() << "Error is: " << e.what();
      onlineDbService_->logger().logError() << "RESULTS OF DEFAULT FIT WAS:";
      onlineDbService_->logger().logError() << "\n" << bs;
      DBloggerReturn_ = -1;
    }
    onlineDbService_->logger().logInfo()
        << "FakeBeamMonitor::FitAndFill - [PayloadCreation] writeForNextLumisection executed correctly";
  }
  edm::LogInfo("FakeBeamMonitor") << "FitAndFill::[PayloadCreation] BeamSpotOnline payload created \n" << std::endl;

  //    }       //if (theBeamFitter->runPVandTrkFitter())
  //    else {  // beam fit fails
  //      reco::BeamSpot bs = theBeamFitter->getBeamSpot();
  //      edm::LogInfo("BeamMonitor") << "FitAndFill::   [BeamMonitor] Beam fit fails!!! \n" << endl;
  //      edm::LogInfo("BeamMonitor") << "FitAndFill::   [BeamMonitor] Output beam spot for DIP \n" << endl;
  //      edm::LogInfo("BeamMonitor") << bs << endl;
  //
  //      hs[k_sigmaX0_lumi]->ShiftFillLast(bs.BeamWidthX(), bs.BeamWidthXError(), fitNLumi_);
  //      hs[k_sigmaY0_lumi]->ShiftFillLast(bs.BeamWidthY(), bs.BeamWidthYError(), fitNLumi_);
  //      hs[k_sigmaZ0_lumi]->ShiftFillLast(bs.sigmaZ(), bs.sigmaZ0Error(), fitNLumi_);
  //      hs[k_x0_lumi]->ShiftFillLast(bs.x0(), bs.x0Error(), fitNLumi_);
  //      hs[k_y0_lumi]->ShiftFillLast(bs.y0(), bs.y0Error(), fitNLumi_);
  //      hs[k_z0_lumi]->ShiftFillLast(bs.z0(), bs.z0Error(), fitNLumi_);
  //    }  // end of beam fit fails

  //  }       //-------- end of countFitting------------------------------------------
  //  else {  // no fit
  //    // Overwrite Fit LS and fit time when no event processed or no track selected
  //    theBeamFitter->setFitLSRange(beginLumiOfBSFit_, endLumiOfBSFit_);
  //    theBeamFitter->setRefTime(refBStime[0], refBStime[1]);
  //    if (theBeamFitter->runPVandTrkFitter()) {
  //    }  // Dump fake beam spot for DIP
  //    reco::BeamSpot bs = theBeamFitter->getBeamSpot();
  //    edm::LogInfo("BeamMonitor") << "FitAndFill::  [BeamMonitor] No fitting \n" << endl;
  //    edm::LogInfo("BeamMonitor") << "FitAndFill::  [BeamMonitor] Output fake beam spot for DIP \n" << endl;
  //    edm::LogInfo("BeamMonitor") << bs << endl;
  //
  //    hs[k_sigmaX0_lumi]->ShiftFillLast(bs.BeamWidthX(), bs.BeamWidthXError(), fitNLumi_);
  //    hs[k_sigmaY0_lumi]->ShiftFillLast(bs.BeamWidthY(), bs.BeamWidthYError(), fitNLumi_);
  //    hs[k_sigmaZ0_lumi]->ShiftFillLast(bs.sigmaZ(), bs.sigmaZ0Error(), fitNLumi_);
  //    hs[k_x0_lumi]->ShiftFillLast(bs.x0(), bs.x0Error(), fitNLumi_);
  //    hs[k_y0_lumi]->ShiftFillLast(bs.y0(), bs.y0Error(), fitNLumi_);
  //    hs[k_z0_lumi]->ShiftFillLast(bs.z0(), bs.z0Error(), fitNLumi_);
  //  }

  // Fill summary report
  //  if (countFitting) {
  for (int n = 0; n < nFitElements_; n++) {
    reportSummaryContents[n]->Fill(summaryContent_[n] / (float)nFits_);
  }

  summarySum_ = 0;
  for (int ii = 0; ii < nFitElements_; ii++) {
    summarySum_ += summaryContent_[ii];
  }
  reportSummary_ = summarySum_ / (nFitElements_ * nFits_);
  if (reportSummary)
    reportSummary->Fill(reportSummary_);

  for (int bi = 0; bi < nFitElements_; bi++) {
    reportSummaryMap->setBinContent(1, bi + 1, summaryContent_[bi] / (float)nFits_);
  }
  //  }

  if ((resetFitNLumi_ > 0 &&
       ((onlineMode_ &&
         countLumi_ == resetFitNLumi_) ||  //OR it should be currentLumi_ (if in sequence then does not mattar)
        (!onlineMode_ && countLumi_ == resetFitNLumi_))) ||
      (StartAverage_)) {
    edm::LogInfo("FakeBeamMonitor") << "FitAndFill:: The flag is ON for running average Beam Spot fit" << endl;
    StartAverage_ = true;
    firstAverageFit_++;
    resetHistos_ = true;
    nthBSTrk_ = 0;
    beginLumiOfBSFit_ = 0;
    refBStime[0] = 0;
  }
}

//--------------------------------------------------------
void FakeBeamMonitor::RestartFitting() {
  if (debug_)
    edm::LogInfo("FakeBeamMonitor")
        << " RestartingFitting:: Restart Beami everything to a fresh start !!! because Gap is > 10 LS" << endl;
  //track based fit reset here
  resetHistos_ = true;
  nthBSTrk_ = 0;
  //  theBeamFitter->resetTrkVector();
  //  theBeamFitter->resetLSRange();
  //  theBeamFitter->resetRefTime();
  //  theBeamFitter->resetPVFitter();
  //  theBeamFitter->resetCutFlow();
  beginLumiOfBSFit_ = 0;
  refBStime[0] = 0;
  //pv based fit iis reset here
  h_PVx[0]->Reset();
  h_PVy[0]->Reset();
  h_PVz[0]->Reset();
  beginLumiOfPVFit_ = 0;
  refPVtime[0] = 0;
  //Clear all the Maps here
  mapPVx.clear();
  mapPVy.clear();
  mapPVz.clear();
  mapNPV.clear();
  mapBeginBSLS.clear();
  mapBeginPVLS.clear();
  mapBeginBSTime.clear();
  mapBeginPVTime.clear();
  mapLSBSTrkSize.clear();
  mapLSPVStoreSize.clear();
  mapLSCF.clear();
  countGapLumi_ = 0;
  countLumi_ = 0;
  StartAverage_ = false;
}

//-------------------------------------------------------
void FakeBeamMonitor::dqmEndRun(const Run& r, const EventSetup& context) {
  if (debug_)
    edm::LogInfo("FakeBeamMonitor") << "dqmEndRun:: Clearing all the Maps " << endl;
  //Clear all the Maps here
  mapPVx.clear();
  mapPVy.clear();
  mapPVz.clear();
  mapNPV.clear();
  mapBeginBSLS.clear();
  mapBeginPVLS.clear();
  mapBeginBSTime.clear();
  mapBeginPVTime.clear();
  mapLSBSTrkSize.clear();
  mapLSPVStoreSize.clear();
  mapLSCF.clear();
}

//--------------------------------------------------------
void FakeBeamMonitor::scrollTH1(TH1* h, time_t ref) {
  char offsetTime[64];
  formatFitTime(offsetTime, ref);
  TDatime da(offsetTime);
  if (lastNZbin > 0) {
    double val = h->GetBinContent(lastNZbin);
    double valErr = h->GetBinError(lastNZbin);
    h->Reset();
    h->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
    int bin = (lastNZbin > buffTime ? buffTime : 1);
    h->SetBinContent(bin, val);
    h->SetBinError(bin, valErr);
  } else {
    h->Reset();
    h->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  }
}

//--------------------------------------------------------
// Method to check whether to chane histogram time offset (forward only)
bool FakeBeamMonitor::testScroll(time_t& tmpTime_, time_t& refTime_) {
  bool scroll_ = false;
  if (tmpTime_ - refTime_ >= intervalInSec_) {
    scroll_ = true;
    edm::LogInfo("FakeBeamMonitor") << "testScroll::  Reset Time Offset" << std::endl;
    lastNZbin = intervalInSec_;
    for (int bin = intervalInSec_; bin >= 1; bin--) {
      if (hs[k_x0_time]->getBinContent(bin) > 0) {
        lastNZbin = bin;
        break;
      }
    }
    edm::LogInfo("FakeBeamMonitor") << "testScroll::  Last non zero bin = " << lastNZbin << std::endl;
    if (tmpTime_ - refTime_ >= intervalInSec_ + lastNZbin) {
      edm::LogInfo("FakeBeamMonitor") << "testScroll::  Time difference too large since last readout" << std::endl;
      lastNZbin = 0;
      refTime_ = tmpTime_ - buffTime;
    } else {
      edm::LogInfo("FakeBeamMonitor") << "testScroll::  Offset to last record" << std::endl;
      int offset = ((lastNZbin > buffTime) ? (lastNZbin - buffTime) : (lastNZbin - 1));
      refTime_ += offset;
    }
  }
  return scroll_;
}

DEFINE_FWK_MODULE(FakeBeamMonitor);

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
