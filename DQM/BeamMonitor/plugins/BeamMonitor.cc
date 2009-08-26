/*
 * \file BeamMonitor.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 * $Date: 2009/08/25 21:46:56 $
 * $Revision: 1.1 $
 *
 */

#include "DQM/BeamMonitor/interface/BeamMonitor.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <numeric>
#include <math.h>
#include <TMath.h>
#include <iostream>
#include "TStyle.h"

using namespace std;
using namespace edm;

//
// constructors and destructor
//
BeamMonitor::BeamMonitor( const ParameterSet& ps ) :
  countEvt_(0),countLumi_(0),nthBSTrk_(0),resetHistos_(false)
{
  parameters_     = ps;
  monitorName_    = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  bsSrc_          = parameters_.getUntrackedParameter<string>("beamSpot","offlineBeamSpot");
  fitNLumi_       = parameters_.getUntrackedParameter<int>("fitEveryNLumi",-1);
  resetFitNLumi_  = parameters_.getUntrackedParameter<int>("resetEveryNLumi",-1);
  debug_          = parameters_.getUntrackedParameter<bool>("Debug");

  dbe_            = Service<DQMStore>().operator->();
  
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  
  theBeamFitter = new BeamFitter(parameters_);
  theBeamFitter->resetTrkVector();

  //   fBSvector.clear();

}


BeamMonitor::~BeamMonitor()
{
  delete theBeamFitter;
}


//--------------------------------------------------------
void BeamMonitor::beginJob(const EventSetup& context){
  
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
  
  // create and cd into new folder
  dbe_->setCurrentFolder(monitorName_+"Fit");
  
  h_nTrk_lumi=dbe_->book1D("nTrk_lumi","Num. of input tracks vs lumi",20,0.5,10.5);
  h_nTrk_lumi->setAxisTitle("Lumisection",1);
  h_nTrk_lumi->setAxisTitle("Num of Tracks",2);
 
  h_d0_phi0 = dbe_->bookProfile("d0_phi0","d_{0} vs. #phi_{0} (Input Tracks)",phiBin,phiMin,phiMax,dxBin,dxMin,dxMax,"");
  h_d0_phi0->setAxisTitle("#phi_{0} (rad)",1);
  h_d0_phi0->setAxisTitle("d_{0} (cm)",2);
 
  h_vx_vy = dbe_->book2D("trk_vx_vy","Vertex (PCA) position of input tracks",vxBin,vxMin,vxMax,vxBin,vxMin,vxMax);
  h_vx_vy->getTH2F()->SetOption("COLZ");
  //   h_vx_vy->getTH1()->SetBit(TH1::kCanRebin);
  h_vx_vy->setAxisTitle("x coordinate of input track at PCA (cm)",1);
  h_vx_vy->setAxisTitle("y coordinate of input track at PCA (cm)",2);
  
  h_x0_lumi = dbe_->book1D("x0_lumi","x coordinate of beam spot vs lumi (Fit)",10,0,10);
  h_x0_lumi->setAxisTitle("Lumisection",1);
  h_x0_lumi->setAxisTitle("x_{0} (cm)",2);
  h_x0_lumi->getTH1()->SetOption("E1");

  h_y0_lumi = dbe_->book1D("y0_lumi","y coordinate of beam spot vs lumi (Fit)",10,0,10);
  h_y0_lumi->setAxisTitle("Lumisection",1);
  h_y0_lumi->setAxisTitle("y_{0} (cm)",2);
  h_y0_lumi->getTH1()->SetOption("E1");
  
}

//--------------------------------------------------------
void BeamMonitor::beginRun(const edm::Run& r, const EventSetup& context) {
  
}

//--------------------------------------------------------
void BeamMonitor::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				       const EventSetup& context) {
  countLumi_++;
  if (debug_) cout << "Lumi: " << countLumi_ << endl;
}

// ----------------------------------------------------------
void BeamMonitor::analyze(const Event& iEvent, 
			  const EventSetup& iSetup )
{  
  countEvt_++;
  theBeamFitter->readEvent(iEvent);
}


//--------------------------------------------------------
void BeamMonitor::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				     const EventSetup& iSetup) {

  vector<BSTrkParameters> theBSvector = theBeamFitter->getBSvector();
  h_nTrk_lumi->ShiftFillLast( theBSvector.size() );
  
  if (fitNLumi_ > 0 && countLumi_%fitNLumi_!=0) return;
  
  if (resetHistos_) {
    h_d0_phi0->Reset();
    h_vx_vy->Reset();
    resetHistos_ = false;
  }
  
  if (debug_) cout << "Fill histos, start from " << nthBSTrk_ + 1 << "th record of input tracks" << endl;
  int i = 0;
  for (vector<BSTrkParameters>::const_iterator BSTrk = theBSvector.begin();
       BSTrk != theBSvector.end();
       ++BSTrk, ++i){
    if (i >= nthBSTrk_){
      h_d0_phi0->Fill( BSTrk->phi0(), BSTrk->d0() );
      double vx = BSTrk->d0()*sin(BSTrk->phi0());
      double vy = -1.*BSTrk->d0()*cos(BSTrk->phi0());
      h_vx_vy->Fill( vx, vy );
    }
  }
  nthBSTrk_ = theBSvector.size();
  if (debug_) cout << "Num of tracks collected = " << nthBSTrk_ << endl;

  if (theBeamFitter->runFitter()){
    reco::BeamSpot bs = theBeamFitter->getBeamSpot();
    if (debug_) {
      cout << "\n RESULTS OF DEFAULT FIT:" << endl;
      cout << bs << endl;
      cout << "[BeamFitter] fitting done \n" << endl;
    }
    h_x0_lumi->ShiftFillLast( bs.x0(), bs.x0Error(), fitNLumi_ );
    h_y0_lumi->ShiftFillLast( bs.y0(), bs.y0Error(), fitNLumi_ );
  }
  if (resetFitNLumi_ > 0 && countLumi_%resetFitNLumi_ == 0) {
    if (debug_) cout << "Reset track collection for beam fit!!!" <<endl;
    resetHistos_ = true;
    nthBSTrk_ = 0;
    theBeamFitter->resetTrkVector();
  }
}
//--------------------------------------------------------
void BeamMonitor::endRun(const Run& r, const EventSetup& context){
  
  
}
//--------------------------------------------------------
void BeamMonitor::endJob(){

}

DEFINE_FWK_MODULE(BeamMonitor);
