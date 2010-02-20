/*
 * \file BeamMonitor.cc
 * \author Geng-yuan Jeng/UC Riverside
 *         Francisco Yumiceva/FNAL
 * $Date: 2010/02/11 00:00:36 $
 * $Revision: 1.22 $
 *
 */

#include "DQM/BeamMonitor/interface/BeamMonitor.h"
#include "DQMServices/Core/interface/QReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <numeric>
#include <math.h>
#include <TMath.h>
#include <iostream>

using namespace std;
using namespace edm;

//
// constructors and destructor
//
BeamMonitor::BeamMonitor( const ParameterSet& ps ) :
  countEvt_(0),countLumi_(0),nthBSTrk_(0),nFitElements_(3),resetHistos_(false) {

  parameters_     = ps;
  monitorName_    = parameters_.getUntrackedParameter<string>("monitorName","YourSubsystemName");
  bsSrc_          = parameters_.getUntrackedParameter<InputTag>("beamSpot");
  pvSrc_          = parameters_.getUntrackedParameter<InputTag>("primaryVertex");
  fitNLumi_       = parameters_.getUntrackedParameter<int>("fitEveryNLumi",-1);
  resetFitNLumi_  = parameters_.getUntrackedParameter<int>("resetEveryNLumi",-1);
  fitPVNLumi_     = parameters_.getUntrackedParameter<int>("fitPVEveryNLumi",-1);
  resetPVNLumi_   = parameters_.getUntrackedParameter<int>("resetPVEveryNLumi",-1);
  deltaSigCut_    = parameters_.getUntrackedParameter<double>("deltaSignificanceCut",15);
  debug_          = parameters_.getUntrackedParameter<bool>("Debug");
  tracksLabel_    = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<InputTag>("TrackCollection");
  min_Ntrks_      = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<int>("MinimumInputTracks");
  maxZ_           = parameters_.getParameter<ParameterSet>("BeamFitter").getUntrackedParameter<double>("MaximumZ");

  dbe_            = Service<DQMStore>().operator->();
  
  if (monitorName_ != "" ) monitorName_ = monitorName_+"/" ;
  
  theBeamFitter = new BeamFitter(parameters_);
  theBeamFitter->resetTrkVector();
  if (fitNLumi_ <= 0) fitNLumi_ = 1;
  nFits_ = beginLumiOfPVFit_ = endLumiOfPVFit_ = 0;
  maxZ_ = fabs(maxZ_);
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
  
  h_x0_lumi = dbe_->book1D("x0_lumi","x coordinate of beam spot vs lumi (Fit)",40,0.5,40.5);
  h_x0_lumi->setAxisTitle("Lumisection",1);
  h_x0_lumi->setAxisTitle("x_{0} (cm)",2);
  h_x0_lumi->getTH1()->SetOption("E1");
  
  h_y0_lumi = dbe_->book1D("y0_lumi","y coordinate of beam spot vs lumi (Fit)",40,0.5,40.5);
  h_y0_lumi->setAxisTitle("Lumisection",1);
  h_y0_lumi->setAxisTitle("y_{0} (cm)",2);
  h_y0_lumi->getTH1()->SetOption("E1");
  
  h_z0_lumi = dbe_->book1D("z0_lumi","z coordinate of beam spot vs lumi (Fit)",40,0.5,40.5);
  h_z0_lumi->setAxisTitle("Lumisection",1);
  h_z0_lumi->setAxisTitle("z_{0} (cm)",2);
  h_z0_lumi->getTH1()->SetOption("E1");
  
  h_sigmaZ0_lumi = dbe_->book1D("sigmaZ0_lumi","sigma z_{0} of beam spot vs lumi (Fit)",40,0.5,40.5);
  h_sigmaZ0_lumi->setAxisTitle("Lumisection",1);
  h_sigmaZ0_lumi->setAxisTitle("sigma z_{0} (cm)",2);
  h_sigmaZ0_lumi->getTH1()->SetOption("E1");
  
  h_trk_z0 = dbe_->book1D("trk_z0","z_{0} of selected tracks",dzBin,dzMin,dzMax);
  h_trk_z0->setAxisTitle("z_{0} of selected tracks (cm)",1);

  h_vx_dz = dbe_->bookProfile("vx_dz","v_{x} vs. dz of selected tracks",dzBin,dzMin,dzMax,dxBin,dxMin,dxMax,"");
  h_vx_dz->setAxisTitle("dz (cm)",1);
  h_vx_dz->setAxisTitle("x coordinate of input track at PCA (cm)",2);

  h_vy_dz = dbe_->bookProfile("vy_dz","v_{y} vs. dz of selected tracks",dzBin,dzMin,dzMax,dxBin,dxMin,dxMax,"");
  h_vy_dz->setAxisTitle("dz (cm)",1);
  h_vy_dz->setAxisTitle("x coordinate of input track at PCA (cm)",2);

  h_x0 = dbe_->book1D("x0","x coordinate of beam spot (Fit)",100,-0.01,0.01);
  h_x0->setAxisTitle("x_{0} (cm)",1);
  h_x0->getTH1()->SetBit(TH1::kCanRebin);

  h_y0 = dbe_->book1D("y0","y coordinate of beam spot (Fit)",100,-0.01,0.01);
  h_y0->setAxisTitle("y_{0} (cm)",1);
  h_y0->getTH1()->SetBit(TH1::kCanRebin);

  h_z0 = dbe_->book1D("z0","z coordinate of beam spot (Fit)",dzBin,dzMin,dzMax);
  h_z0->setAxisTitle("z_{0} (cm)",1);
  h_z0->getTH1()->SetBit(TH1::kCanRebin);

  h_sigmaZ0 = dbe_->book1D("sigmaZ0","sigma z0 of beam spot (Fit)",100,0,10);
  h_sigmaZ0->setAxisTitle("sigmaZ_{0} (cm)",1);
  h_sigmaZ0->getTH1()->SetBit(TH1::kCanRebin);

  // Histograms of all reco tracks (without cuts):
  h_trkPt=dbe_->book1D("trkPt","p_{T} of all reco'd tracks (no selection)",200,0.,50.);
  h_trkPt->setAxisTitle("p_{T} (GeV/c)",1);

  h_trkVz=dbe_->book1D("trkVz","Z coordinate of PCA of all reco'd tracks (no selection)",dzBin,dzMin,dzMax);
  h_trkVz->setAxisTitle("V_{Z} (cm)",1);

  cutFlowTable = dbe_->book1D("cutFlowTable","Cut flow table of track selection", 9, 0, 9 );

  // Results of previous good fit:
  fitResults=dbe_->book2D("fitResults","Results of previous good fit",2,0,2,4,0,4);
  fitResults->setAxisTitle("Fitted Beam Spot (cm)",1);
  fitResults->setBinLabel(4,"x_{0}",2);
  fitResults->setBinLabel(3,"y_{0}",2);
  fitResults->setBinLabel(2,"z_{0}",2);
  fitResults->setBinLabel(1,"#sigma_{Z0}",2);
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

  h_PVx_lumi = dbe_->book1D("PVx_lumi","Avg. x position of primary vtx vs lumi",40,0.5,40.5);
  h_PVx_lumi->setAxisTitle("Lumisection",1);
  h_PVx_lumi->setAxisTitle("PVx #pm #sigma_{PVx} (cm)",2);
  h_PVx_lumi->getTH1()->SetOption("E1");

  h_PVy_lumi = dbe_->book1D("PVy_lumi","Avg. y position of primary vtx vs lumi",40,0.5,40.5);
  h_PVy_lumi->setAxisTitle("Lumisection",1);
  h_PVy_lumi->setAxisTitle("PVy #pm #sigma_{PVy} (cm)",2);
  h_PVy_lumi->getTH1()->SetOption("E1");

  h_PVz_lumi = dbe_->book1D("PVz_lumi","Avg. z position of primary vtx vs lumi",40,0.5,40.5);
  h_PVz_lumi->setAxisTitle("Lumisection",1);
  h_PVz_lumi->setAxisTitle("PVz #pm #sigma_{PVz} (cm)",2);
  h_PVz_lumi->getTH1()->SetOption("E1");

  h_PVxz = dbe_->bookProfile("PVxz","PVx vs. PVz",dzBin/2,dzMin,dzMax,dxBin/2,dxMin,dxMax,"");
  h_PVxz->setAxisTitle("PVz (cm)",1);
  h_PVxz->setAxisTitle("PVx (cm)",2);

  h_PVyz = dbe_->bookProfile("PVyz","PVy vs. PVz",dzBin/2,dzMin,dzMax,dxBin/2,dxMin,dxMax,"");
  h_PVyz->setAxisTitle("PVz (cm)",1);
  h_PVyz->setAxisTitle("PVy (cm)",2);

  // Results of previous good fit:
  pvResults=dbe_->book2D("pvResults","Results of avg. PV positions",2,0,2,3,0,3);
  pvResults->setAxisTitle("Fitted Primary Vertex (cm)",1);
  pvResults->setBinLabel(3,"PVx",2);
  pvResults->setBinLabel(2,"PVy",2);
  pvResults->setBinLabel(1,"PVz",2);
  pvResults->setBinLabel(1,"Mean",1);
  pvResults->setBinLabel(2,"Width",1);
  pvResults->getTH1()->SetOption("text");

  // Summary plots:
  dbe_->setCurrentFolder(monitorName_+"EventInfo");
  reportSummary = dbe_->get(monitorName_+"EventInfo/reportSummary");
  if (reportSummary) dbe_->removeElement(reportSummary->getName());

  reportSummary = dbe_->bookFloat("reportSummary");
  if(reportSummary) reportSummary->Fill(-1.);

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
    reportSummaryContents[i]->Fill(-1.);
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
  
}

//--------------------------------------------------------
void BeamMonitor::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				       const EventSetup& context) {
  if (resetPVNLumi_ > 0 && countLumi_%resetPVNLumi_ == 0)
    beginLumiOfPVFit_ = lumiSeg.luminosityBlock();

  countLumi_++;
  if (debug_) cout << "Lumi: " << countLumi_ << endl;
}

// ----------------------------------------------------------
void BeamMonitor::analyze(const Event& iEvent, 
			  const EventSetup& iSetup ) {  
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

  edm::Handle< edm::View<reco::Vertex> > recVtxs;
  iEvent.getByLabel(pvSrc_, recVtxs);
  const edm::View<reco::Vertex> &pv = *recVtxs;

  int nGoodPV=0;
  int idx=0;
  double sum=0.0;

  for (size_t ipv = 0; ipv != pv.size(); ++ipv) {
    double sum_ntracks=0.0;
    int ntracks=0;

    if (pv[ipv].isFake()) continue;
    h_nVtx->Fill(recVtxs->size()*1.);
    for (reco::Vertex::trackRef_iterator it = pv[ipv].tracks_begin(); it != pv[ipv].tracks_end(); it++) {
      sum+= pv[ipv].trackWeight(*it);
      ntracks++;
    }

    if(ntracks>0){sum_ntracks=(sum/(float)ntracks);}
    else{sum_ntracks=0.0;}
    
    if (sum_ntracks > 0.5 && pv[ipv].ndof() > 2) {
      nGoodPV++;
      idx=ipv;
      if (debug_) std::cout << "Which PV passes selection? # " << idx << std::endl;
    }
  }

  if (nGoodPV == 1) { // Temporary, select only events contain one good PV
    if (debug_) std::cout << "Which PV is selected? # " << idx << std::endl;
    h_PVx[0]->Fill(pv[idx].x());
    h_PVy[0]->Fill(pv[idx].y());
    h_PVz[0]->Fill(pv[idx].z());
    h_PVxz->Fill(pv[idx].z(),pv[idx].x());
    h_PVyz->Fill(pv[idx].z(),pv[idx].y());
  }

}


//--------------------------------------------------------
void BeamMonitor::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				     const EventSetup& iSetup) {

  if (fitPVNLumi_ > 0 && countLumi_%fitPVNLumi_ == 0) {
    endLumiOfPVFit_ = lumiSeg.luminosityBlock();
    // Primary Vertex Fit:
    if (h_PVx[0]->getTH1()->GetEntries() >= 20) {
      pvResults->Reset();
      char tmpTitle[50];
      sprintf(tmpTitle,"%s %i %s %i","Fitted Primary Vertex (cm) of LS: ",beginLumiOfPVFit_," to ",endLumiOfPVFit_);
      pvResults->setAxisTitle(tmpTitle,1);

      TF1 *fgaus = new TF1("fgaus","gaus");
      double mean,width;
      fgaus->SetLineColor(4);
      h_PVx[0]->getTH1()->Fit("fgaus","QLM0");
      mean = fgaus->GetParameter(1);
      width = fgaus->GetParameter(2);
      h_PVx_lumi->ShiftFillLast(mean,width);
      pvResults->setBinContent(1,3,mean);
      pvResults->setBinContent(2,3,width);
    
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
      h_PVy_lumi->ShiftFillLast(mean,width);
      pvResults->setBinContent(1,2,mean);
      pvResults->setBinContent(2,2,width);
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
      h_PVz_lumi->ShiftFillLast(mean,width);
      pvResults->setBinContent(1,1,mean);
      pvResults->setBinContent(2,1,width);
      // snap shot of the fit
      tmpfile= (h_PVz[1]->getName()).c_str();
      tmphisto = static_cast<TH1D *>((h_PVz[0]->getTH1())->Clone("tmphisto"));
      h_PVz[1]->getTH1()->SetBins(tmphisto->GetNbinsX(),tmphisto->GetXaxis()->GetXmin(),tmphisto->GetXaxis()->GetXmax());
      h_PVz[1]->update();
      h_PVz[1] = dbe_->book1D(tmpfile,h_PVz[0]->getTH1F());
      h_PVz[1]->getTH1()->Fit("fgaus","QLM");

    }
  }

  if (resetPVNLumi_ > 0 && countLumi_%resetPVNLumi_ == 0) {
    h_PVx[0]->Reset();
    h_PVy[0]->Reset();
    h_PVz[0]->Reset();
  }

//   if (resetPVNLumi_ > 0 && countLumi_%(4*resetPVNLumi_) == 0) {
//     h_PVxz->Reset();
//     h_PVyz->Reset();
//   }

  // Beam Spot Fit:
  vector<BSTrkParameters> theBSvector = theBeamFitter->getBSvector();
  h_nTrk_lumi->ShiftFillLast( theBSvector.size() );
  
  bool fitted = false;
  if (theBSvector.size() > nthBSTrk_ && theBSvector.size() >= min_Ntrks_) {
    nFits_++;
    fitted = true;
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
  if (debug_ && fitted) cout << "Num of tracks collected = " << nthBSTrk_ << endl;

  if (fitNLumi_ > 0 && countLumi_%fitNLumi_!=0) return;

  if (nthBSTrk_ >= min_Ntrks_) {
    TF1 *f1 = new TF1("f1","[0]*sin(x-[1])",-3.15,3.15);
    f1->SetLineColor(4);
    h_d0_phi0->getTProfile()->Fit("f1","QR");

    TF1 *fgaus = new TF1("fgaus","gaus");
    fgaus->SetLineColor(4);
    h_trk_z0->getTH1()->Fit("fgaus","QLRM","",-1*maxZ_,maxZ_);
  }

  if (fitted && theBeamFitter->runFitter()){
    reco::BeamSpot bs = theBeamFitter->getBeamSpot();
    preBS = bs;
    if (debug_) {
      cout << "\n RESULTS OF DEFAULT FIT:" << endl;
      cout << bs << endl;
      cout << "[BeamFitter] fitting done \n" << endl;
    }
    h_x0_lumi->ShiftFillLast( bs.x0(), bs.x0Error(), fitNLumi_ );
    h_y0_lumi->ShiftFillLast( bs.y0(), bs.y0Error(), fitNLumi_ );
    h_z0_lumi->ShiftFillLast( bs.z0(), bs.z0Error(), fitNLumi_ );
    h_sigmaZ0_lumi->ShiftFillLast( bs.sigmaZ(), bs.sigmaZ0Error(), fitNLumi_ );

    h_x0->Fill( bs.x0());
    h_y0->Fill( bs.y0());
    h_z0->Fill( bs.z0());
    h_sigmaZ0->Fill( bs.sigmaZ());

    fitResults->Reset();
    int * LSRange = theBeamFitter->getFitLSRange();
    char tmpTitle[50];
    sprintf(tmpTitle,"%s %i %s %i","Fitted Beam Spot (cm) of LS: ",LSRange[0]," to ",LSRange[1]);
    fitResults->setAxisTitle(tmpTitle,1);
    fitResults->setBinContent(1,4,bs.x0());
    fitResults->setBinContent(1,3,bs.y0());
    fitResults->setBinContent(1,2,bs.z0());
    fitResults->setBinContent(1,1,bs.sigmaZ());
    fitResults->setBinContent(2,4,bs.x0Error());
    fitResults->setBinContent(2,3,bs.y0Error());
    fitResults->setBinContent(2,2,bs.z0Error());
    fitResults->setBinContent(2,1,bs.sigmaZ0Error());

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
  else { // FIXME: temporarily fill in previous fitted results
    if (debug_) {
      cout << "[BeamFitter] No fitting \n" << endl;
      cout << "Fill previous good fitted results:" << endl;
      cout << preBS << endl;
    }
    h_x0_lumi->ShiftFillLast( preBS.x0(), preBS.x0Error(), fitNLumi_ );
    h_y0_lumi->ShiftFillLast( preBS.y0(), preBS.y0Error(), fitNLumi_ );
    h_z0_lumi->ShiftFillLast( preBS.z0(), preBS.z0Error(), fitNLumi_ );
    h_sigmaZ0_lumi->ShiftFillLast( preBS.sigmaZ(), preBS.sigmaZ0Error(), fitNLumi_ );
  }
  
  // Fill summary report
  if (fitted) {
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

  if (resetFitNLumi_ > 0 && countLumi_%resetFitNLumi_ == 0) {
    if (debug_) cout << "Reset track collection for beam fit!!!" <<endl;
    resetHistos_ = true;
    nthBSTrk_ = 0;
    theBeamFitter->resetTrkVector();
    theBeamFitter->resetLSRange();
  }
}
//--------------------------------------------------------
void BeamMonitor::endRun(const Run& r, const EventSetup& context){

}
//--------------------------------------------------------
void BeamMonitor::endJob(const LuminosityBlock& lumiSeg, 
			 const EventSetup& iSetup){
  endLuminosityBlock(lumiSeg, iSetup);
}

DEFINE_FWK_MODULE(BeamMonitor);
