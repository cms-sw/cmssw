/*
 * \file BeamMonitor.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 * $Date: 2010/04/15 15:29:22 $
 * $Revision: 1.44 $
 *
 */

#include "DQM/BeamMonitor/interface/BeamMonitor.h"
#include "DQMServices/Core/interface/QReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include <numeric>
#include <math.h>
#include <TMath.h>
#include <iostream>
#include <TStyle.h>

using namespace std;
using namespace edm;

static char * formatTime( const time_t t )  {
#define CET (+1)
#define CEST (+2)

  static  char ts[] = "yyyy-Mm-dd hh:mm:ss     ";
  tm * ptm;
  ptm = gmtime ( &t );
  sprintf( ts, "%4d-%02d-%02d %02d:%02d:%02d", ptm->tm_year,ptm->tm_mon+1,ptm->tm_mday,(ptm->tm_hour+CEST)%24, ptm->tm_min, ptm->tm_sec);

#ifdef STRIP_TRAILING_BLANKS_IN_TIMEZONE
  unsigned int b = strlen(ts);
  while (ts[--b] == ' ') {ts[b] = 0;}
#endif
  return ts;

}

#define buffTime (23)

//
// constructors and destructor
//
BeamMonitor::BeamMonitor( const ParameterSet& ps ) :
  countEvt_(0),countLumi_(0),nthBSTrk_(0),nFitElements_(3),resetHistos_(false) {

  parameters_     = ps;
  monitorName_    = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  bsSrc_          = parameters_.getUntrackedParameter<InputTag>("beamSpot");
  pvSrc_          = parameters_.getUntrackedParameter<InputTag>("primaryVertex");
  intervalInSec_  = parameters_.getUntrackedParameter<int>("timeInterval",920);//40 LS X 23"
  fitNLumi_       = parameters_.getUntrackedParameter<int>("fitEveryNLumi",-1);
  resetFitNLumi_  = parameters_.getUntrackedParameter<int>("resetEveryNLumi",-1);
  fitPVNLumi_     = parameters_.getUntrackedParameter<int>("fitPVEveryNLumi",-1);
  resetPVNLumi_   = parameters_.getUntrackedParameter<int>("resetPVEveryNLumi",-1);
  deltaSigCut_    = parameters_.getUntrackedParameter<double>("deltaSignificanceCut",15);
  debug_          = parameters_.getUntrackedParameter<bool>("Debug");
  onlineMode_     = parameters_.getUntrackedParameter<bool>("OnlineMode");
  tracksLabel_    = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<InputTag>("TrackCollection");
  min_Ntrks_      = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumInputTracks");
  maxZ_           = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumZ");
  minNrVertices_  = parameters_.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<unsigned int>("minNrVerticesForFit");
  minVtxNdf_      = parameters_.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexNdf");
  minVtxWgt_      = parameters_.getParameter<ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexMeanWeight");

  dbe_            = Service<DQMStore>().operator->();
  
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  
  theBeamFitter = new BeamFitter(parameters_);
  theBeamFitter->resetTrkVector();
  theBeamFitter->resetLSRange();
  theBeamFitter->resetRefTime();
  theBeamFitter->resetPVFitter();

  if (fitNLumi_ <= 0) fitNLumi_ = 1;
  nFits_ = beginLumiOfBSFit_ = endLumiOfBSFit_ = beginLumiOfPVFit_ = endLumiOfPVFit_ = 0;
  refBStime[0] = refBStime[1] = refPVtime[0] = refPVtime[1] = 0;
  maxZ_ = fabs(maxZ_);
  lastlumi_ = 0;
  nextlumi_ = 0;

}


BeamMonitor::~BeamMonitor() {
  delete theBeamFitter;
}


//--------------------------------------------------------
void BeamMonitor::beginJob() {

  // book some histograms here
  const int    dxBin = parameters_.getParameter<int>("dxBin");
  const double dxMin  = parameters_.getParameter<double>("dxMin");
  const double dxMax  = parameters_.getParameter<double>("dxMax");

  const int    vxBin = parameters_.getParameter<int>("vxBin");
  const double vxMin  = parameters_.getParameter<double>("vxMin");
  const double vxMax  = parameters_.getParameter<double>("vxMax");
  
  const int    phiBin = parameters_.getParameter<int>("phiBin");
  const double phiMin  = parameters_.getParameter<double>("phiMin");
  const double phiMax  = parameters_.getParameter<double>("phiMax");

  const int    dzBin = parameters_.getParameter<int>("dzBin");
  const double dzMin  = parameters_.getParameter<double>("dzMin");
  const double dzMax  = parameters_.getParameter<double>("dzMax");

  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Fit");
  
  h_nTrk_lumi=dbe_->book1D("nTrk_lumi","Num. of selected tracks vs lumi",20,0.5,20.5);
  h_nTrk_lumi->setAxisTitle("Lumisection",1);
  h_nTrk_lumi->setAxisTitle("Num of Tracks",2);
  
  h_d0_phi0 = dbe_->bookProfile("d0_phi0","d_{0} vs. #phi_{0} (Selected Tracks)",phiBin,phiMin,phiMax,dxBin,dxMin,dxMax,"");
  h_d0_phi0->setAxisTitle("#phi_{0} (rad)",1);
  h_d0_phi0->setAxisTitle("d_{0} (cm)",2);
  
  h_vx_vy = dbe_->book2D("trk_vx_vy","Vertex (PCA) position of selected tracks",vxBin,vxMin,vxMax,vxBin,vxMin,vxMax);
  h_vx_vy->getTH2F()->SetOption("COLZ");
  //   h_vx_vy->getTH1()->SetBit(TH1::kCanRebin);
  h_vx_vy->setAxisTitle("x coordinate of input track at PCA (cm)",1);
  h_vx_vy->setAxisTitle("y coordinate of input track at PCA (cm)",2);

  TDatime *da = new TDatime();
  gStyle->SetTimeOffset(da->Convert(kTRUE));

  char coord[] = {"xyz"};
  for (int i = 0; i < 4; i++) {
    dbe_->setCurrentFolder(monitorName_+"Fit");
    for (int ic=0; ic<3; ++ic) {
      string histName(1,coord[ic]);
      string histTitle(1,coord[ic]);
      string ytitle(1,coord[ic]);
      string xtitle("");
      string options("E1");
      switch(i) {
      case 1: // BS vs time
	histName += "0_time";
	histTitle += " coordinate of beam spot vs time (Fit)";
	xtitle += "Time [UTC]";
	ytitle += "_{0} (cm)";
	break;
      case 2: // PV vs lumi
	dbe_->setCurrentFolder(monitorName_+"PrimaryVertex");
	histName.insert(0,"PV");
	histName += "_lumi";
	histTitle.insert(0,"Avg. ");
	histTitle += " position of primary vtx vs lumi";
	xtitle += "Lumisection";
	ytitle.insert(0,"PV");
	ytitle += " #pm #sigma_{PV";
	ytitle += coord[ic];
	ytitle += "} (cm)";
	break;
      case 3: // PV vs time
	dbe_->setCurrentFolder(monitorName_+"PrimaryVertex");
	histName.insert(0,"PV");
	histName += "_time";
	histTitle.insert(0,"Avg. ");
	histTitle += " position of primary vtx vs time";
	xtitle += "Time [UTC]";
	ytitle.insert(0,"PV");
	ytitle += " #pm #sigma_{PV";
	ytitle += coord[ic];
	ytitle += "} (cm)";
	break;
      default: // BS vs lumi
	histName += "0_lumi";
	histTitle += " coordinate of beam spot vs lumi (Fit)";
	xtitle += "Lumisection";
	ytitle += "_{0} (cm)";
	break;
      }
      if (debug_) std::cout << "hitsName = " << histName << "; histTitle = " << histTitle << std::endl;
      hs[histName] = dbe_->book1D(histName,histTitle,40,0.5,40.5);
      hs[histName]->setAxisTitle(xtitle,1);
      hs[histName]->setAxisTitle(ytitle,2);
      hs[histName]->getTH1()->SetOption("E1");
      if (histName.find("time") != string::npos) {
	//int nbins = (intervalInSec_/23 > 0 ? intervalInSec_/23 : 40);
 	hs[histName]->getTH1()->SetBins(intervalInSec_,0.5,intervalInSec_+0.5);
	hs[histName]->setAxisTimeDisplay(1);
	hs[histName]->setAxisTimeFormat("%H:%M:%S",1);
      }
    }
  }
  dbe_->setCurrentFolder(monitorName_+"Fit");

  h_sigmaX0_lumi = dbe_->book1D("sigmaX0_lumi","sigma x_{0} of beam spot vs lumi (Fit)",40,0.5,40.5);
  h_sigmaX0_lumi->setAxisTitle("Lumisection",1);
  h_sigmaX0_lumi->setAxisTitle("#sigma_{X_{0}} (cm)",2);
  h_sigmaX0_lumi->getTH1()->SetOption("E1");

  h_sigmaX0_time = dbe_->book1D("sigmaX0_time","sigma x_{0} of beam spot vs time (Fit)",intervalInSec_,0.5,intervalInSec_+0.5);
  h_sigmaX0_time->setAxisTitle("Time [UTC]",1);
  h_sigmaX0_time->setAxisTitle("#sigma_{X_{0}} (cm)",2);
  h_sigmaX0_time->getTH1()->SetOption("E1");
  h_sigmaX0_time->setAxisTimeDisplay(1);
  h_sigmaX0_time->setAxisTimeFormat("%H:%M:%S",1);

  h_sigmaY0_lumi = dbe_->book1D("sigmaY0_lumi","sigma y_{0} of beam spot vs lumi (Fit)",40,0.5,40.5);
  h_sigmaY0_lumi->setAxisTitle("Lumisection",1);
  h_sigmaY0_lumi->setAxisTitle("#sigma_{Y_{0}} (cm)",2);
  h_sigmaY0_lumi->getTH1()->SetOption("E1");

  h_sigmaY0_time = dbe_->book1D("sigmaY0_time","sigma y_{0} of beam spot vs time (Fit)",intervalInSec_,0.5,intervalInSec_+0.5);
  h_sigmaY0_time->setAxisTitle("Time [UTC]",1);
  h_sigmaY0_time->setAxisTitle("#sigma_{Y_{0}} (cm)",2);
  h_sigmaY0_time->getTH1()->SetOption("E1");
  h_sigmaY0_time->setAxisTimeDisplay(1);
  h_sigmaY0_time->setAxisTimeFormat("%H:%M:%S",1);

  h_sigmaZ0_lumi = dbe_->book1D("sigmaZ0_lumi","sigma z_{0} of beam spot vs lumi (Fit)",40,0.5,40.5);
  h_sigmaZ0_lumi->setAxisTitle("Lumisection",1);
  h_sigmaZ0_lumi->setAxisTitle("#sigma_{Z_{0}} (cm)",2);
  h_sigmaZ0_lumi->getTH1()->SetOption("E1");

  h_sigmaZ0_time = dbe_->book1D("sigmaZ0_time","sigma z_{0} of beam spot vs time (Fit)",intervalInSec_,0.5,intervalInSec_+0.5);
  h_sigmaZ0_time->setAxisTitle("Time [UTC]",1);
  h_sigmaZ0_time->setAxisTitle("#sigma_{Z_{0}} (cm)",2);
  h_sigmaZ0_time->getTH1()->SetOption("E1");
  h_sigmaZ0_time->setAxisTimeDisplay(1);
  h_sigmaZ0_time->setAxisTimeFormat("%H:%M:%S",1);

  h_trk_z0 = dbe_->book1D("trk_z0","z_{0} of selected tracks",dzBin,dzMin,dzMax);
  h_trk_z0->setAxisTitle("z_{0} of selected tracks (cm)",1);

  h_vx_dz = dbe_->bookProfile("vx_dz","v_{x} vs. dz of selected tracks",dzBin,dzMin,dzMax,dxBin,dxMin,dxMax,"");
  h_vx_dz->setAxisTitle("dz (cm)",1);
  h_vx_dz->setAxisTitle("x coordinate of input track at PCA (cm)",2);

  h_vy_dz = dbe_->bookProfile("vy_dz","v_{y} vs. dz of selected tracks",dzBin,dzMin,dzMax,dxBin,dxMin,dxMax,"");
  h_vy_dz->setAxisTitle("dz (cm)",1);
  h_vy_dz->setAxisTitle("x coordinate of input track at PCA (cm)",2);

  h_x0 = dbe_->book1D("BeamMonitorFeedBack_x0","x coordinate of beam spot (Fit)",100,-0.01,0.01);
  h_x0->setAxisTitle("x_{0} (cm)",1);
  h_x0->getTH1()->SetBit(TH1::kCanRebin);

  h_y0 = dbe_->book1D("BeamMonitorFeedBack_y0","y coordinate of beam spot (Fit)",100,-0.01,0.01);
  h_y0->setAxisTitle("y_{0} (cm)",1);
  h_y0->getTH1()->SetBit(TH1::kCanRebin);

  h_z0 = dbe_->book1D("BeamMonitorFeedBack_z0","z coordinate of beam spot (Fit)",dzBin,dzMin,dzMax);
  h_z0->setAxisTitle("z_{0} (cm)",1);
  h_z0->getTH1()->SetBit(TH1::kCanRebin);

  h_sigmaX0 = dbe_->book1D("BeamMonitorFeedBack_sigmaX0","sigma x0 of beam spot (Fit)",100,0,0.05);
  h_sigmaX0->setAxisTitle("#sigma_{X_{0}} (cm)",1);
  h_sigmaX0->getTH1()->SetBit(TH1::kCanRebin);

  h_sigmaY0 = dbe_->book1D("BeamMonitorFeedBack_sigmaY0","sigma y0 of beam spot (Fit)",100,0,0.05);
  h_sigmaY0->setAxisTitle("#sigma_{Y_{0}} (cm)",1);
  h_sigmaY0->getTH1()->SetBit(TH1::kCanRebin);

  h_sigmaZ0 = dbe_->book1D("BeamMonitorFeedBack_sigmaZ0","sigma z0 of beam spot (Fit)",100,0,10);
  h_sigmaZ0->setAxisTitle("#sigma_{Z_{0}} (cm)",1);
  h_sigmaZ0->getTH1()->SetBit(TH1::kCanRebin);

  // Histograms of all reco tracks (without cuts):
  h_trkPt=dbe_->book1D("trkPt","p_{T} of all reco'd tracks (no selection)",200,0.,50.);
  h_trkPt->setAxisTitle("p_{T} (GeV/c)",1);

  h_trkVz=dbe_->book1D("trkVz","Z coordinate of PCA of all reco'd tracks (no selection)",dzBin,dzMin,dzMax);
  h_trkVz->setAxisTitle("V_{Z} (cm)",1);

  cutFlowTable = dbe_->book1D("cutFlowTable","Cut flow table of track selection", 9, 0, 9 );

  // Results of previous good fit:
  fitResults=dbe_->book2D("fitResults","Results of previous good beam fit",2,0,2,8,0,8);
  fitResults->setAxisTitle("Fitted Beam Spot (cm)",1);
  fitResults->setBinLabel(8,"x_{0}",2);
  fitResults->setBinLabel(7,"y_{0}",2);
  fitResults->setBinLabel(6,"z_{0}",2);
  fitResults->setBinLabel(5,"#sigma_{Z}",2);
  fitResults->setBinLabel(4,"#frac{dx}{dz} (rad)",2);
  fitResults->setBinLabel(3,"#frac{dy}{dz} (rad)",2);
  fitResults->setBinLabel(2,"#sigma_{X}",2);
  fitResults->setBinLabel(1,"#sigma_{Y}",2);
  fitResults->setBinLabel(1,"Mean",1);
  fitResults->setBinLabel(2,"Stat. Error",1);
  fitResults->getTH1()->SetOption("text");

  // Histos of PrimaryVertices:
  dbe_->setCurrentFolder(monitorName_+"PrimaryVertex");

  h_nVtx = dbe_->book1D("vtxNbr","Reconstructed Vertices in Event",20,-0.5,19.5);
  h_nVtx->setAxisTitle("Num. of reco. vertices",1);

  // Monitor only the PV with highest sum pt of assoc. trks:
  h_PVx[0] = dbe_->book1D("PVX","x coordinate of Primary Vtx",50,-0.01,0.01);
  h_PVx[0]->setAxisTitle("PVx (cm)",1);
  h_PVx[0]->getTH1()->SetBit(TH1::kCanRebin);

  h_PVy[0] = dbe_->book1D("PVY","y coordinate of Primary Vtx",50,-0.01,0.01);
  h_PVy[0]->setAxisTitle("PVy (cm)",1);
  h_PVy[0]->getTH1()->SetBit(TH1::kCanRebin);

  h_PVz[0] = dbe_->book1D("PVZ","z coordinate of Primary Vtx",dzBin,dzMin,dzMax);
  h_PVz[0]->setAxisTitle("PVz (cm)",1);

  h_PVx[1] = dbe_->book1D("PVXFit","x coordinate of Primary Vtx (Last Fit)",50,-0.01,0.01);
  h_PVx[1]->setAxisTitle("PVx (cm)",1);
  h_PVx[1]->getTH1()->SetBit(TH1::kCanRebin);

  h_PVy[1] = dbe_->book1D("PVYFit","y coordinate of Primary Vtx (Last Fit)",50,-0.01,0.01);
  h_PVy[1]->setAxisTitle("PVy (cm)",1);
  h_PVy[1]->getTH1()->SetBit(TH1::kCanRebin);

  h_PVz[1] = dbe_->book1D("PVZFit","z coordinate of Primary Vtx (Last Fit)",dzBin,dzMin,dzMax);
  h_PVz[1]->setAxisTitle("PVz (cm)",1);

  h_PVxz = dbe_->bookProfile("PVxz","PVx vs. PVz",dzBin/2,dzMin,dzMax,dxBin/2,dxMin,dxMax,"");
  h_PVxz->setAxisTitle("PVz (cm)",1);
  h_PVxz->setAxisTitle("PVx (cm)",2);

  h_PVyz = dbe_->bookProfile("PVyz","PVy vs. PVz",dzBin/2,dzMin,dzMax,dxBin/2,dxMin,dxMax,"");
  h_PVyz->setAxisTitle("PVz (cm)",1);
  h_PVyz->setAxisTitle("PVy (cm)",2);

  // Results of previous good fit:
  pvResults=dbe_->book2D("pvResults","Results of fitting Primary Vertices",2,0,2,6,0,6);
  pvResults->setAxisTitle("Fitted Primary Vertex (cm)",1);
  pvResults->setBinLabel(6,"PVx",2);
  pvResults->setBinLabel(5,"PVy",2);
  pvResults->setBinLabel(4,"PVz",2);
  pvResults->setBinLabel(3,"#sigma_{X}",2);
  pvResults->setBinLabel(2,"#sigma_{Y}",2);
  pvResults->setBinLabel(1,"#sigma_{Z}",2);
  pvResults->setBinLabel(1,"Mean",1);
  pvResults->setBinLabel(2,"Stat. Error",1);
  pvResults->getTH1()->SetOption("text");

  // Summary plots:
  dbe_->setCurrentFolder(monitorName_+"EventInfo");
  reportSummary = dbe_->get(monitorName_+"EventInfo/reportSummary");
  if (reportSummary) dbe_->removeElement(reportSummary->getName());

  reportSummary = dbe_->bookFloat("reportSummary");
  if(reportSummary) reportSummary->Fill(0./0.);

  char histo[20];
  dbe_->setCurrentFolder(monitorName_+"EventInfo/reportSummaryContents");
  for (int n = 0; n < nFitElements_; n++) {
    switch(n){
    case 0 : sprintf(histo,"x0_status"); break;
    case 1 : sprintf(histo,"y0_status"); break;
    case 2 : sprintf(histo,"z0_status"); break;
    }
    reportSummaryContents[n] = dbe_->bookFloat(histo);
  }

  for (int i = 0; i < nFitElements_; i++) {
    summaryContent_[i] = 0.;
    reportSummaryContents[i]->Fill(0./0.);
  }
  
  dbe_->setCurrentFolder(monitorName_+"EventInfo");

  reportSummaryMap = dbe_->get(monitorName_+"EventInfo/reportSummaryMap");
  if (reportSummaryMap) dbe_->removeElement(reportSummaryMap->getName());
  
  reportSummaryMap = dbe_->book2D("reportSummaryMap", "Beam Spot Summary Map", 1, 0, 1, 3, 0, 3);
  reportSummaryMap->setAxisTitle("",1);
  reportSummaryMap->setAxisTitle("Fitted Beam Spot",2);
  reportSummaryMap->setBinLabel(1," ",1);
  reportSummaryMap->setBinLabel(1,"x_{0}",2);
  reportSummaryMap->setBinLabel(2,"y_{0}",2);
  reportSummaryMap->setBinLabel(3,"z_{0}",2);
  for (int i = 0; i < nFitElements_; i++) {
    reportSummaryMap->setBinContent(1,i+1,-1.);
  }
}

//--------------------------------------------------------
void BeamMonitor::beginRun(const edm::Run& r, const EventSetup& context) {

  ftimestamp = r.beginTime().value();
  tmpTime = ftimestamp >> 32;
  refTime =  tmpTime;
  char* eventTime = formatTime(tmpTime);
  TDatime da(eventTime);
  if (debug_) {
    std::cout << "TimeOffset = ";
    da.Print();
  }
  hs["x0_time"]->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  hs["y0_time"]->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  hs["z0_time"]->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  h_sigmaX0_time->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  h_sigmaY0_time->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  h_sigmaZ0_time->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  hs["PVx_time"]->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  hs["PVy_time"]->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  hs["PVz_time"]->getTH1()->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));

}

//--------------------------------------------------------
void BeamMonitor::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				       const EventSetup& context) {
  int nthlumi = lumiSeg.luminosityBlock();
  const edm::TimeValue_t fbegintimestamp = lumiSeg.beginTime().value();
  const std::time_t ftmptime = fbegintimestamp >> 32;

  if (countLumi_ == 0) {
    beginLumiOfBSFit_ = beginLumiOfPVFit_ = nthlumi;
    refBStime[0] = refPVtime[0] = ftmptime;
  }
  if (beginLumiOfPVFit_ == 0) beginLumiOfPVFit_ = nextlumi_;
  if (beginLumiOfBSFit_ == 0) beginLumiOfBSFit_ = nextlumi_;

  if (onlineMode_ && (nthlumi < nextlumi_)) return;

  if (onlineMode_) {
    if (nthlumi > nextlumi_) {
      if (countLumi_ != 0) FitAndFill(lumiSeg,lastlumi_,nextlumi_,nthlumi);
      nextlumi_ = nthlumi;
      cout << "Next Lumi to Fit: " << nextlumi_ << endl;
      if (refBStime[0] == 0) refBStime[0] = ftmptime;
      if (refPVtime[0] == 0) refPVtime[0] = ftmptime;
    }
  }
  else{
    FitAndFill(lumiSeg,lastlumi_,nextlumi_,nthlumi);
    if (refBStime[0] == 0) refBStime[0] = ftmptime;
    if (refPVtime[0] == 0) refPVtime[0] = ftmptime;
  }
  countLumi_++;
  //if (debug_)
  cout << "Begin of Lumi: " << nthlumi << endl;
}

// ----------------------------------------------------------
void BeamMonitor::analyze(const Event& iEvent, 
			  const EventSetup& iSetup ) {
  bool readEvent_ = true;
  const int nthlumi = iEvent.luminosityBlock();
  if (onlineMode_ && (nthlumi < nextlumi_)) {
    readEvent_ = false;
    if (debug_) std::cout << "Spilt event from previous lumi section!" << std::endl;
  }
  if (onlineMode_ && (nthlumi > nextlumi_)) {
    readEvent_ = false;
    if (debug_) std::cout << "Spilt event from next lumi section!!!" << std::endl;
  }

  if (readEvent_) {
    countEvt_++;
    theBeamFitter->readEvent(iEvent);
    Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByLabel(bsSrc_,recoBeamSpotHandle);
    refBS = *recoBeamSpotHandle;

    dbe_->setCurrentFolder(monitorName_+"Fit/");
    const char* tmpfile;
    TH1F * tmphisto;
    tmpfile = (cutFlowTable->getName()).c_str();
    tmphisto = static_cast<TH1F *>((theBeamFitter->getCutFlow())->Clone("tmphisto"));
    cutFlowTable->getTH1()->SetBins(tmphisto->GetNbinsX(),tmphisto->GetXaxis()->GetXmin(),tmphisto->GetXaxis()->GetXmax());
    if (countEvt_ == 1) // SetLabel just once
      for(int n=0; n < tmphisto->GetNbinsX(); n++)
	cutFlowTable->setBinLabel(n+1,tmphisto->GetXaxis()->GetBinLabel(n+1),1);

    cutFlowTable = dbe_->book1D(tmpfile,tmphisto);

    Handle<reco::TrackCollection> TrackCollection;
    iEvent.getByLabel(tracksLabel_, TrackCollection);
    const reco::TrackCollection *tracks = TrackCollection.product();
    for ( reco::TrackCollection::const_iterator track = tracks->begin();
	  track != tracks->end(); ++track ) {    
      h_trkPt->Fill(track->pt());
      h_trkVz->Fill(track->vz());
    }

    //------ Primary Vertices
    edm::Handle< reco::VertexCollection > PVCollection;

    if (iEvent.getByLabel(pvSrc_, PVCollection )) {
      int nPVcount = 0;
      for (reco::VertexCollection::const_iterator pv = PVCollection->begin(); pv != PVCollection->end(); ++pv) {
	//--- vertex selection
	if (pv->isFake() || pv->tracksSize()==0)  continue;
	nPVcount++; // count non fake pv
	if (pv->ndof() < minVtxNdf_ || (pv->ndof()+3.)/pv->tracksSize() < 2*minVtxWgt_)  continue;
	h_PVx[0]->Fill(pv->x());
	h_PVy[0]->Fill(pv->y());
	h_PVz[0]->Fill(pv->z());
	h_PVxz->Fill(pv->z(),pv->x());
	h_PVyz->Fill(pv->z(),pv->y());
      }
      if (nPVcount>0) h_nVtx->Fill(nPVcount*1.);
    }
  }//end of read event

}


//--------------------------------------------------------
void BeamMonitor::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				     const EventSetup& iSetup) {
  int nthlumi = lumiSeg.id().luminosityBlock();
  //if (debug_)
  cout << "Lumi of the last event before endLuminosityBlock: " << nthlumi << endl;

  if (onlineMode_ && nthlumi < nextlumi_) return;
  const edm::TimeValue_t fendtimestamp = lumiSeg.endTime().value();
  const std::time_t fendtime = fendtimestamp >> 32;
  tmpTime = refBStime[1] = refPVtime[1] = fendtime;
}

//--------------------------------------------------------
void BeamMonitor::FitAndFill(const LuminosityBlock& lumiSeg,int &lastlumi,int &nextlumi,int &nthlumi){
  if (onlineMode_ && (nthlumi <= nextlumi)) return;

  int currentlumi = nextlumi;
  cout << "Lumi of the current fit: " << currentlumi << endl;
  lastlumi = currentlumi;
  endLumiOfBSFit_ = currentlumi;
  endLumiOfPVFit_ = currentlumi;

  if (onlineMode_) { // filling LS gap
    // FIXME: need to add protection for the case if the gap is at the resetting LS!
    const int countLS_bs = hs["x0_lumi"]->getTH1()->GetEntries();
    const int countLS_pv = hs["PVx_lumi"]->getTH1()->GetEntries();
    if (debug_) std::cout << "countLS_bs = " << countLS_bs << " ; countLS_pv = " << countLS_pv << std::endl;
    int LSgap_bs = currentlumi/fitNLumi_ - countLS_bs;
    int LSgap_pv = currentlumi/fitPVNLumi_ - countLS_pv;
    if (currentlumi%fitNLumi_ == 0)
      LSgap_bs--;
    if (currentlumi%fitPVNLumi_ == 0)
      LSgap_pv--;
    if (debug_) std::cout << "LSgap_bs = " << LSgap_bs << " ; LSgap_pv = " << LSgap_pv << std::endl;
    // filling previous fits if LS gap ever exists
    for (int ig = 0; ig < LSgap_bs; ig++) {
      hs["x0_lumi"]->ShiftFillLast( 0., 0., fitNLumi_ );
      hs["y0_lumi"]->ShiftFillLast( 0., 0., fitNLumi_ );
      hs["z0_lumi"]->ShiftFillLast( 0., 0., fitNLumi_ );
      h_sigmaX0_lumi->ShiftFillLast( 0., 0., fitNLumi_ );
      h_sigmaY0_lumi->ShiftFillLast( 0., 0., fitNLumi_ );
      h_sigmaZ0_lumi->ShiftFillLast( 0., 0., fitNLumi_ );
    }
    for (int ig = 0; ig < LSgap_pv; ig++) {
      hs["PVx_lumi"]->ShiftFillLast( 0., 0., fitPVNLumi_ );
      hs["PVy_lumi"]->ShiftFillLast( 0., 0., fitPVNLumi_ );
      hs["PVz_lumi"]->ShiftFillLast( 0., 0., fitPVNLumi_ );
    }
    const int previousLS = h_nTrk_lumi->getTH1()->GetEntries();
    for (int i=1;i < (currentlumi - previousLS);i++)
      h_nTrk_lumi->ShiftFillLast(nthBSTrk_);
  }
  else { // not online mode
    h_nTrk_lumi->ShiftFillLast(nthBSTrk_);
  }

  if (debug_) std::cout << "Time lapsed since last LS = " << tmpTime - refTime << std:: endl;

  if (testScroll(tmpTime,refTime)) {
    scrollTH1(hs["x0_time"]->getTH1(),refTime);
    scrollTH1(hs["y0_time"]->getTH1(),refTime);
    scrollTH1(hs["z0_time"]->getTH1(),refTime);
    scrollTH1(h_sigmaX0_time->getTH1(),refTime);
    scrollTH1(h_sigmaY0_time->getTH1(),refTime);
    scrollTH1(h_sigmaZ0_time->getTH1(),refTime);
    scrollTH1(hs["PVx_time"]->getTH1(),refTime);
    scrollTH1(hs["PVy_time"]->getTH1(),refTime);
    scrollTH1(hs["PVz_time"]->getTH1(),refTime);
  }

  bool doPVFit = false;

  if (fitPVNLumi_ > 0) {
    if (onlineMode_) {
      if (currentlumi%fitPVNLumi_ == 0)
	doPVFit = true;
    }
    else
      if (countLumi_%fitPVNLumi_ == 0)
	doPVFit = true;
  }
  else
    doPVFit = true;

  if (doPVFit) {

    if (debug_) std::cout << "Do PV Fitting for LS = " << beginLumiOfPVFit_ << " to " << endLumiOfPVFit_ << std::endl;
    // Primary Vertex Fit:
    if (h_PVx[0]->getTH1()->GetEntries() > minNrVertices_) {
      pvResults->Reset();
      char tmpTitle[50];
      sprintf(tmpTitle,"%s %i %s %i","Fitted Primary Vertex (cm) of LS: ",beginLumiOfPVFit_," to ",endLumiOfPVFit_);
      pvResults->setAxisTitle(tmpTitle,1);

      TF1 *fgaus = new TF1("fgaus","gaus");
      double mean,width,meanErr,widthErr;
      fgaus->SetLineColor(4);
      h_PVx[0]->getTH1()->Fit("fgaus","QLM0");
      mean = fgaus->GetParameter(1);
      width = fgaus->GetParameter(2);
      meanErr = fgaus->GetParError(1);
      widthErr = fgaus->GetParError(2);
      hs["PVx_lumi"]->ShiftFillLast(mean,width,fitPVNLumi_);
      int nthBin = tmpTime - refTime;
      if (nthBin < 0 && debug_)
	std::cout << "Event time outside current range of time histograms!" << std::endl;
      if (nthBin > 0) {
	hs["PVx_time"]->setBinContent(nthBin,mean);
	hs["PVx_time"]->setBinError(nthBin,width);
      }
      pvResults->setBinContent(1,6,mean);
      pvResults->setBinContent(1,3,width);
      pvResults->setBinContent(2,6,meanErr);
      pvResults->setBinContent(2,3,widthErr);
    
      dbe_->setCurrentFolder(monitorName_+"PrimaryVertex/");
      const char* tmpfile;
      TH1D * tmphisto;
      // snap shot of the fit
      tmpfile= (h_PVx[1]->getName()).c_str();
      tmphisto = static_cast<TH1D *>((h_PVx[0]->getTH1())->Clone("tmphisto"));
      h_PVx[1]->getTH1()->SetBins(tmphisto->GetNbinsX(),tmphisto->GetXaxis()->GetXmin(),tmphisto->GetXaxis()->GetXmax());
      h_PVx[1] = dbe_->book1D(tmpfile,h_PVx[0]->getTH1F());
      h_PVx[1]->getTH1()->Fit("fgaus","QLM");


      h_PVy[0]->getTH1()->Fit("fgaus","QLM0");
      mean = fgaus->GetParameter(1);
      width = fgaus->GetParameter(2);
      meanErr = fgaus->GetParError(1);
      widthErr = fgaus->GetParError(2);
      hs["PVy_lumi"]->ShiftFillLast(mean,width,fitPVNLumi_);
      if (nthBin > 0) {
	hs["PVy_time"]->setBinContent(nthBin,mean);
	hs["PVy_time"]->setBinError(nthBin,width);
      }
      pvResults->setBinContent(1,5,mean);
      pvResults->setBinContent(1,2,width);
      pvResults->setBinContent(2,5,meanErr);
      pvResults->setBinContent(2,2,widthErr);
      // snap shot of the fit
      tmpfile= (h_PVy[1]->getName()).c_str();
      tmphisto = static_cast<TH1D *>((h_PVy[0]->getTH1())->Clone("tmphisto"));
      h_PVy[1]->getTH1()->SetBins(tmphisto->GetNbinsX(),tmphisto->GetXaxis()->GetXmin(),tmphisto->GetXaxis()->GetXmax());
      h_PVy[1]->update();
      h_PVy[1] = dbe_->book1D(tmpfile,h_PVy[0]->getTH1F());
      h_PVy[1]->getTH1()->Fit("fgaus","QLM");


      h_PVz[0]->getTH1()->Fit("fgaus","QLM0");
      mean = fgaus->GetParameter(1);
      width = fgaus->GetParameter(2);
      meanErr = fgaus->GetParError(1);
      widthErr = fgaus->GetParError(2);
      hs["PVz_lumi"]->ShiftFillLast(mean,width,fitPVNLumi_);
      if (nthBin > 0) {
	hs["PVz_time"]->setBinContent(nthBin,mean);
	hs["PVz_time"]->setBinError(nthBin,width);
      }
      pvResults->setBinContent(1,4,mean);
      pvResults->setBinContent(1,1,width);
      pvResults->setBinContent(2,4,meanErr);
      pvResults->setBinContent(2,1,widthErr);
      // snap shot of the fit
      tmpfile= (h_PVz[1]->getName()).c_str();
      tmphisto = static_cast<TH1D *>((h_PVz[0]->getTH1())->Clone("tmphisto"));
      h_PVz[1]->getTH1()->SetBins(tmphisto->GetNbinsX(),tmphisto->GetXaxis()->GetXmin(),tmphisto->GetXaxis()->GetXmax());
      h_PVz[1]->update();
      h_PVz[1] = dbe_->book1D(tmpfile,h_PVz[0]->getTH1F());
      h_PVz[1]->getTH1()->Fit("fgaus","QLM");

    }
  }

  if (resetPVNLumi_ > 0 &&
      ((onlineMode_ && currentlumi%resetPVNLumi_ == 0) ||
       (!onlineMode_ && countLumi_%resetPVNLumi_ == 0))) {
    h_PVx[0]->Reset();
    h_PVy[0]->Reset();
    h_PVz[0]->Reset();
    beginLumiOfPVFit_ = 0;
    refPVtime[0] = 0;
  }

  // Beam Spot Fit:
  vector<BSTrkParameters> theBSvector = theBeamFitter->getBSvector();
  h_nTrk_lumi->ShiftFillLast( theBSvector.size() );
  
  bool doFitting = false;
  if (theBSvector.size() > nthBSTrk_ && theBSvector.size() >= min_Ntrks_) {
    doFitting = true;
  }

  if (resetHistos_) {
    if (debug_) cout << "Resetting Histograms" << endl;
    h_d0_phi0->Reset();
    h_vx_vy->Reset();
    h_vx_dz->Reset();
    h_vy_dz->Reset();
    h_trk_z0->Reset();
    theBeamFitter->resetCutFlow();
    resetHistos_ = false;
  }
  
  if (debug_) cout << "Fill histos, start from " << nthBSTrk_ + 1 << "th record of selected tracks" << endl;
  unsigned int itrk = 0;
  for (vector<BSTrkParameters>::const_iterator BSTrk = theBSvector.begin();
       BSTrk != theBSvector.end();
       ++BSTrk, ++itrk){
    if (itrk >= nthBSTrk_){
      h_d0_phi0->Fill( BSTrk->phi0(), BSTrk->d0() );
      double vx = BSTrk->vx();
      double vy = BSTrk->vy();
      double z0 = BSTrk->z0();
      h_vx_vy->Fill( vx, vy );
      h_vx_dz->Fill( z0, vx );
      h_vy_dz->Fill( z0, vy );
      h_trk_z0->Fill( z0 );
    }
  }
  nthBSTrk_ = theBSvector.size(); // keep track of num of tracks filled so far
  if (debug_ && doFitting) cout << "Num of tracks collected = " << nthBSTrk_ << endl;

  if (fitNLumi_ > 0) {
    if (onlineMode_){
      if (currentlumi%fitNLumi_!=0) return;
    }
    else      
      if (countLumi_%fitNLumi_!=0) return;
  }

  if (doFitting) {
    nFits_++;
    int * fitLS = theBeamFitter->getFitLSRange();
    if (debug_) std::cout << "[BeamFitter] Do BeamSpot Fit for LS = " << fitLS[0] << " to " << fitLS[1] << std::endl;
    if (debug_) std::cout << "[BeamMonitor] Do BeamSpot Fit for LS = " << beginLumiOfBSFit_ << " to " << endLumiOfBSFit_ << std::endl;
    //theBeamFitter->setFitLSRange(beginLumiOfBSFit_,endLumiOfBSFit_); // Testing, overwrite Fit LS
    //theBeamFitter->setRefTime(refBStime[0],refBStime[1]); // Testing, overwrite Fit time
  }

  if (doFitting && theBeamFitter->runFitter()){
    reco::BeamSpot bs = theBeamFitter->getBeamSpot();
    if (bs.type() > 0) // with good beamwidth fit
      preBS = bs; // cache good fit results
    if (debug_) {
      cout << "\n RESULTS OF DEFAULT FIT:" << endl;
      cout << bs << endl;
      cout << "[BeamFitter] fitting done \n" << endl;
    }
    hs["x0_lumi"]->ShiftFillLast( bs.x0(), bs.x0Error(), fitNLumi_ );
    hs["y0_lumi"]->ShiftFillLast( bs.y0(), bs.y0Error(), fitNLumi_ );
    hs["z0_lumi"]->ShiftFillLast( bs.z0(), bs.z0Error(), fitNLumi_ );
    h_sigmaX0_lumi->ShiftFillLast( bs.BeamWidthX(), bs.BeamWidthXError(), fitNLumi_ );
    h_sigmaY0_lumi->ShiftFillLast( bs.BeamWidthY(), bs.BeamWidthYError(), fitNLumi_ );
    h_sigmaZ0_lumi->ShiftFillLast( bs.sigmaZ(), bs.sigmaZ0Error(), fitNLumi_ );

    int nthBin = tmpTime - refTime;
    if (nthBin > 0) {
      hs["x0_time"]->setBinContent(nthBin, bs.x0());
      hs["y0_time"]->setBinContent(nthBin, bs.y0());
      hs["z0_time"]->setBinContent(nthBin, bs.z0());
      h_sigmaX0_time->setBinContent(nthBin, bs.BeamWidthX());
      h_sigmaY0_time->setBinContent(nthBin, bs.BeamWidthY());
      h_sigmaZ0_time->setBinContent(nthBin, bs.sigmaZ());
      hs["x0_time"]->setBinError(nthBin, bs.x0Error());
      hs["y0_time"]->setBinError(nthBin, bs.y0Error());
      hs["z0_time"]->setBinError(nthBin, bs.z0Error());
      h_sigmaX0_time->setBinError(nthBin, bs.BeamWidthXError());
      h_sigmaY0_time->setBinError(nthBin, bs.BeamWidthYError());
      h_sigmaZ0_time->setBinError(nthBin, bs.sigmaZ0Error());
    }
    h_x0->Fill( bs.x0());
    h_y0->Fill( bs.y0());
    h_z0->Fill( bs.z0());
    if (bs.type() > 0) { // with good beamwidth fit
      h_sigmaX0->Fill( bs.BeamWidthX());
      h_sigmaY0->Fill( bs.BeamWidthY());
    }
    h_sigmaZ0->Fill( bs.sigmaZ());

    if (nthBSTrk_ >= 2*min_Ntrks_) {
      double amp = sqrt(bs.x0()*bs.x0()+bs.y0()*bs.y0());
      double alpha = atan2(bs.y0(),bs.x0());
      TF1 *f1 = new TF1("f1","[0]*sin(x-[1])",-3.14,3.14);
      f1->SetParameters(amp,alpha);
      f1->SetParLimits(1,amp-0.1,amp+0.1);
      f1->SetParLimits(2,alpha-0.577,alpha+0.577);
      f1->SetLineColor(4);
      h_d0_phi0->getTProfile()->Fit("f1","QR");

      double mean = bs.z0();
      double width = bs.sigmaZ();
      TF1 *fgaus = new TF1("fgaus","gaus");
      fgaus->SetParameters(mean,width);
      fgaus->SetLineColor(4);
      h_trk_z0->getTH1()->Fit("fgaus","QLRM","",mean-3*width,mean+3*width);
    }

    fitResults->Reset();
    int * LSRange = theBeamFitter->getFitLSRange();
    char tmpTitle[50];
    sprintf(tmpTitle,"%s %i %s %i","Fitted Beam Spot (cm) of LS: ",LSRange[0]," to ",LSRange[1]);
    fitResults->setAxisTitle(tmpTitle,1);
    fitResults->setBinContent(1,8,bs.x0());
    fitResults->setBinContent(1,7,bs.y0());
    fitResults->setBinContent(1,6,bs.z0());
    fitResults->setBinContent(1,5,bs.sigmaZ());
    fitResults->setBinContent(1,4,bs.dxdz());
    fitResults->setBinContent(1,3,bs.dydz());
    if (bs.type() > 0) { // with good beamwidth fit
      fitResults->setBinContent(1,2,bs.BeamWidthX());
      fitResults->setBinContent(1,1,bs.BeamWidthY());
    }
    else { // fill cached widths
      fitResults->setBinContent(1,2,preBS.BeamWidthX());
      fitResults->setBinContent(1,1,preBS.BeamWidthY());
    }

    fitResults->setBinContent(2,8,bs.x0Error());
    fitResults->setBinContent(2,7,bs.y0Error());
    fitResults->setBinContent(2,6,bs.z0Error());
    fitResults->setBinContent(2,5,bs.sigmaZ0Error());
    fitResults->setBinContent(2,4,bs.dxdzError());
    fitResults->setBinContent(2,3,bs.dydzError());
    if (bs.type() > 0) { // with good beamwidth fit
      fitResults->setBinContent(2,2,bs.BeamWidthXError());
      fitResults->setBinContent(2,1,bs.BeamWidthYError());
    }
    else { // fill cached width errors
      fitResults->setBinContent(2,2,preBS.BeamWidthXError());
      fitResults->setBinContent(2,1,preBS.BeamWidthYError());
    }

//     if (fabs(refBS.x0()-bs.x0())/bs.x0Error() < deltaSigCut_) { // disabled temporarily
      summaryContent_[0] += 1.;
//     }
//     if (fabs(refBS.y0()-bs.y0())/bs.y0Error() < deltaSigCut_) { // disabled temporarily
      summaryContent_[1] += 1.;
//     }
//     if (fabs(refBS.z0()-bs.z0())/bs.z0Error() < deltaSigCut_) { // disabled temporarily
      summaryContent_[2] += 1.;
//     }
  }
  else {
    if (!doFitting) {
      // Overwrite Fit LS and fit time when no event processed or no track selected
      theBeamFitter->setFitLSRange(beginLumiOfBSFit_,endLumiOfBSFit_);
      theBeamFitter->setRefTime(refBStime[0],refBStime[1]);
    }
    if (theBeamFitter->runFitter()) {} // Dump fake beam spot for DIP
    reco::BeamSpot bs = theBeamFitter->getBeamSpot();
    if (debug_) {
      cout << "[BeamMonitor] No fitting \n" << endl;
      cout << "[BeamMonitor] Output fake beam spot for DIP \n" << endl;
      cout << bs << endl;
//       cout << "[BeamMonitor] Fill histograms with previous fitted results:" << endl;
//       if (nFits_ == 0) preBS.setType(reco::BeamSpot::Fake);
//       cout << preBS << endl;
    }
    h_sigmaX0_lumi->ShiftFillLast( bs.BeamWidthX(), bs.BeamWidthXError(), fitNLumi_ );
    h_sigmaY0_lumi->ShiftFillLast( bs.BeamWidthY(), bs.BeamWidthYError(), fitNLumi_ );
    h_sigmaZ0_lumi->ShiftFillLast( bs.sigmaZ(), bs.sigmaZ0Error(), fitNLumi_ );
    hs["x0_lumi"]->ShiftFillLast( bs.x0(), bs.x0Error(), fitNLumi_ );
    hs["y0_lumi"]->ShiftFillLast( bs.y0(), bs.y0Error(), fitNLumi_ );
    hs["z0_lumi"]->ShiftFillLast( bs.z0(), bs.z0Error(), fitNLumi_ );
//     h_sigmaX0_lumi->ShiftFillLast( preBS.BeamWidthX(), preBS.BeamWidthXError(), fitNLumi_ );
//     h_sigmaY0_lumi->ShiftFillLast( preBS.BeamWidthY(), preBS.BeamWidthYError(), fitNLumi_ );
//     h_sigmaZ0_lumi->ShiftFillLast( preBS.sigmaZ(), preBS.sigmaZ0Error(), fitNLumi_ );
//     hs["x0_lumi"]->ShiftFillLast( preBS.x0(), preBS.x0Error(), fitNLumi_ );
//     hs["y0_lumi"]->ShiftFillLast( preBS.y0(), preBS.y0Error(), fitNLumi_ );
//     hs["z0_lumi"]->ShiftFillLast( preBS.z0(), preBS.z0Error(), fitNLumi_ );
  }
  
  // Fill summary report
  if (doFitting) {
    for (int n = 0; n < nFitElements_; n++) {
      reportSummaryContents[n]->Fill( summaryContent_[n] / (float)nFits_ );
    }

    summarySum_ = 0;
    for (int ii = 0; ii < nFitElements_; ii++) {
      summarySum_ += summaryContent_[ii];
    }
    reportSummary_ = summarySum_ / (nFitElements_ * nFits_);
    if (reportSummary) reportSummary->Fill(reportSummary_);

    for ( int bi = 0; bi < nFitElements_ ; bi++) {
      reportSummaryMap->setBinContent(1,bi+1,summaryContent_[bi] / (float)nFits_);
    }
  }

  if (resetFitNLumi_ > 0 && 
      ((onlineMode_ && currentlumi%resetFitNLumi_ == 0) ||
       (!onlineMode_ && countLumi_%resetFitNLumi_ == 0))) {
    if (debug_) cout << "Reset track collection for beam fit!!!" <<endl;
    resetHistos_ = true;
    nthBSTrk_ = 0;
    theBeamFitter->resetTrkVector();
    theBeamFitter->resetLSRange();
    theBeamFitter->resetRefTime();
    theBeamFitter->resetPVFitter();
    beginLumiOfBSFit_ = 0;
    refBStime[0] = 0;
  }
}

//--------------------------------------------------------
void BeamMonitor::endRun(const Run& r, const EventSetup& context){

}

//--------------------------------------------------------
void BeamMonitor::endJob(const LuminosityBlock& lumiSeg, 
			 const EventSetup& iSetup){
  if (!onlineMode_) endLuminosityBlock(lumiSeg, iSetup);
}

//--------------------------------------------------------
void BeamMonitor::scrollTH1(TH1 * h, time_t ref) {
  char* offsetTime = formatTime(ref);
  TDatime da(offsetTime);
  if (lastNZbin > 0) {
    double val = h->GetBinContent(lastNZbin);
    double valErr = h->GetBinError(lastNZbin);
    h->Reset();
    h->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
    int bin = (lastNZbin > buffTime ? buffTime : 1);
    h->SetBinContent(bin,val);
    h->SetBinError(bin,valErr);
  }
  else {
    h->Reset();
    h->GetXaxis()->SetTimeOffset(da.Convert(kTRUE));
  }
}

//--------------------------------------------------------
// Method to check whether to chane histogram time offset (forward only)
bool BeamMonitor::testScroll(time_t & tmpTime_, time_t & refTime_){
  bool scroll_ = false;
  if (tmpTime_ - refTime_ >= intervalInSec_) {
    scroll_ = true;
    if (debug_) std::cout << "Reset Time Offset" << std::endl;
    lastNZbin = intervalInSec_;
    for (int bin = intervalInSec_; bin >= 1; bin--) {
      if (hs["x0_time"]->getBinContent(bin) > 0) {
	lastNZbin = bin;
	break;
      }
    }
    if (debug_) std::cout << "Last non zero bin = " << lastNZbin << std::endl;
    if (tmpTime_ - refTime_ >= intervalInSec_ + lastNZbin) {
      if (debug_) std::cout << "Time difference too large since last readout" << std::endl;
      lastNZbin = 0;
      refTime_ = tmpTime_ - buffTime;
    }
    else{
      if (debug_) std::cout << "Offset to last record" << std::endl;
      int offset = ((lastNZbin > buffTime) ? (lastNZbin - buffTime) : (lastNZbin - 1));
      refTime_ += offset;
    }
  }
  return scroll_;
}

DEFINE_FWK_MODULE(BeamMonitor);
