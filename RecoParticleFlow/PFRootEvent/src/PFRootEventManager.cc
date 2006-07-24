#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "RecoParticleFlow/PFClusterAlgo/interface/PFClusterAlgo.h"
#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/IO.h"

#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TMarker.h>
#include <TH2F.h>
#include <TCutG.h>
#include <TPolyLine.h>
#include <TColor.h>
#include "TGraph.h"
#include "TMath.h"
#include "TLine.h"
#include "TLatex.h"
#include "TVector3.h"

#include <iostream>

using namespace std;

PFRootEventManager::PFRootEventManager() 
{
  graphTrack_.resize(NViews);
}

PFRootEventManager::~PFRootEventManager() 
{
  // Clear TGraph if needed
  for (unsigned iView = 0; iView < graphTrack_.size(); iView++) {
    if (graphTrack_[iView].size()) {
      for (unsigned iGraph = 0; iGraph < graphTrack_[iView].size(); iGraph++)
	delete graphTrack_[iView][iGraph];
      graphTrack_[iView].clear();
    }
  }
}

PFRootEventManager::PFRootEventManager(const char* file) {  
  options_ = 0;
  ReadOptions(file);
  iEvent_=0;
  displayHistEtaPhi_=0;
  displayView_.resize(NViews);
  displayHist_.resize(NViews);
  for (unsigned iView = 0; iView < NViews; iView++) {
    displayView_[iView] = 0;
    displayHist_[iView] = 0;
  }
  graphTrack_.resize(NViews);
  maxERecHitEcal_ = -1;
  maxERecHitHcal_ = -1;
}

void PFRootEventManager::Reset() { 
  maxERecHitEcal_ = -1;
  maxERecHitHcal_ = -1;  
  clusters_.clear();
  rechits_.clear();
  recTracks_.clear();
}

void PFRootEventManager::ReadOptions(const char* file, bool refresh) {

  if( !options_ )
    options_ = new IO(file);
  else if( refresh) {
    delete options_;
    options_ = new IO(file);
  }

  options_->GetOpt("root","file", inFileName_);
  
  file_ = TFile::Open(inFileName_.c_str() );
  if(file_->IsZombie() ) {
    return;
  }

  tree_ = (TTree*) file_->Get("Events");
  
  string hitbranchname;
  options_->GetOpt("root","hits_branch", hitbranchname);
  
  hitsBranch_ = tree_->GetBranch(hitbranchname.c_str());
  if(!hitsBranch_) {
    cerr<<"PFRootEventManager::ReadOptions : branch not found : "
	<<hitbranchname<<endl;
  }
  else {
    hitsBranch_->SetAddress(&rechits_);
    // cout << "Set branch " << hitbranchname << endl;
  }

  string recTracksbranchname;
  options_->GetOpt("root","recTracks_branch", recTracksbranchname);

  recTracksBranch_ = tree_->GetBranch(recTracksbranchname.c_str());
  if(!recTracksBranch_) {
    cerr <<"PFRootEventManager::ReadOptions : branch not found : "
	 << recTracksbranchname << endl;
  }
  else {
    recTracksBranch_->SetAddress(&recTracks_);
    // cout << "Set branch " << recTracksbranchname << " " << recTracksBranch_->GetEntries() << endl;
  }    

  vector<int> algos;
  options_->GetOpt("display", "algos", algos);
  algosToDisplay_.clear();
  for(unsigned i=0; i< algos.size(); i++) algosToDisplay_.insert( algos[i] );
  typedef map<int, TCanvas * >::iterator IT;
  for(IT it = displayEtaPhi_.begin(); it!=displayEtaPhi_.end(); it++) {
    if( algosToDisplay_.find(it->first) == algosToDisplay_.end() ) {
      it->second->Close();
      displayEtaPhi_.erase(it);
      if(displayEtaPhi_.empty() ) break;
      it++;
    }    
  }  

  viewSizeEtaPhi_.clear();
  options_->GetOpt("display", "viewsize_etaphi", viewSizeEtaPhi_);
  if(viewSizeEtaPhi_.size() != 2) {
    cerr<<"PFRootEventManager::ReadOptions, bad display/viewsize_etaphi tag...using 700/350"
	<<endl;
    viewSizeEtaPhi_.clear();
    viewSizeEtaPhi_.push_back(700); 
    viewSizeEtaPhi_.push_back(350); 
  }

  viewSize_.clear();
  options_->GetOpt("display", "viewsize_xy", viewSize_);
  if(viewSize_.size() != 2) {
    cerr<<"PFRootEventManager::ReadOptions, bad display/viewsize_xy tag...using 700/350"
	<<endl;
    viewSize_.clear();
    viewSize_.push_back(600); 
    viewSize_.push_back(600); 
  }

  threshEcalBarrel_ = 0.1;
  options_->GetOpt("clustering", "thresh_Ecal_Barrel", threshEcalBarrel_);
  
  threshSeedEcalBarrel_ = 0.3;
  options_->GetOpt("clustering", "thresh_Seed_Ecal_Barrel", 
		   threshSeedEcalBarrel_);

  threshEcalEndcap_ = 0.2;
  options_->GetOpt("clustering", "thresh_Ecal_Endcap", threshEcalEndcap_);

  threshSeedEcalEndcap_ = 0.8;
  options_->GetOpt("clustering", "thresh_Seed_Ecal_Endcap",
		   threshSeedEcalEndcap_);


  nNeighboursEcal_ = 4;
  options_->GetOpt("clustering", "neighbours_Ecal", nNeighboursEcal_);


  threshHcalBarrel_ = 0.1;
  options_->GetOpt("clustering", "thresh_Hcal_Barrel", threshHcalBarrel_);
  
  threshSeedHcalBarrel_ = 0.3;
  options_->GetOpt("clustering", "thresh_Seed_Hcal_Barrel", 
		   threshSeedHcalBarrel_);

  threshHcalEndcap_ = 0.2;
  options_->GetOpt("clustering", "thresh_Hcal_Endcap", threshHcalEndcap_);

  threshSeedHcalEndcap_ = 0.8;
  options_->GetOpt("clustering", "thresh_Seed_Hcal_Endcap",
		   threshSeedHcalEndcap_);

  nNeighboursHcal_ = 4;
  options_->GetOpt("clustering", "neighbours_Hcal", nNeighboursHcal_);
}


bool PFRootEventManager::ProcessEntry(int entry) {

  Reset();

  cout<<"process entry "<< entry << endl;
  
  if(hitsBranch_) hitsBranch_->GetEntry(entry);
  if(recTracksBranch_) recTracksBranch_->GetEntry(entry);

  cout<<"number of rechits : "<<rechits_.size()<<endl;
  cout<<"number of recTracks : "<<recTracks_.size()<<endl;
//   for(unsigned i=0; i<rechits_.size(); i++) {
//     cout<<rechits_[i]<<endl;
//   }

  Clustering(); 

  return false;
}


void PFRootEventManager::Clustering() {
  
  std::map<unsigned,  reco::PFRecHit* > rechits;
   
  for(unsigned i=0; i<rechits_.size(); i++) {
    rechits.insert( make_pair(rechits_[i].GetDetId(), &rechits_[i] ) );
  }

  for( PFClusterAlgo::IDH ih = rechits.begin(); ih != rechits.end(); ih++) {
    ih->second->FindPtrsToNeighbours( rechits );
  }

  PFClusterAlgo clusteralgo; 

  clusteralgo.SetThreshEcalBarrel( threshEcalBarrel_ );
  clusteralgo.SetThreshSeedEcalBarrel( threshSeedEcalBarrel_ );
  
  clusteralgo.SetThreshEcalEndcap( threshEcalEndcap_ );
  clusteralgo.SetThreshSeedEcalEndcap( threshSeedEcalEndcap_ );

  clusteralgo.SetNNeighboursEcal( nNeighboursEcal_  );
  
  clusteralgo.SetThreshHcalBarrel( threshHcalBarrel_ );
  clusteralgo.SetThreshSeedHcalBarrel( threshSeedHcalBarrel_ );
  
  clusteralgo.SetThreshHcalEndcap( threshHcalEndcap_ );
  clusteralgo.SetThreshSeedHcalEndcap( threshSeedHcalEndcap_ );

  clusteralgo.SetNNeighboursHcal( nNeighboursHcal_ );


  clusteralgo.Init( rechits ); 
  clusteralgo.AllClusters();

  clusters_ = clusteralgo.GetClusters();
}


void PFRootEventManager::Display(int ientry) {
  
  ProcessEntry(ientry);
  DisplayEtaPhi();
  DisplayView(RZ);
  DisplayView(XY);
}


void PFRootEventManager::DisplayEtaPhi() {

  if(!displayHistEtaPhi_) {
    displayHistEtaPhi_ = new TH2F("fHist","",100,-5,5,100,-3.2,3.2);
    displayHistEtaPhi_->SetXTitle("Eta");
    displayHistEtaPhi_->SetYTitle("Phi");
    displayHistEtaPhi_->SetStats(0);
  }

  // clear all cluster displays
  typedef map<int, TCanvas * >::iterator IT;
  for(IT it = displayEtaPhi_.begin(); it!=displayEtaPhi_.end(); it++) {
    TCanvas *c = it->second;
    c->cd(1);
    gPad->Clear();
    c->cd(2);
    gPad->Clear();
  } 

  char cnam[20];
  sprintf(cnam, "test_display_%d", 0);
  TCanvas* c = new TCanvas(cnam, 
			   cnam,
			   viewSizeEtaPhi_[0], viewSizeEtaPhi_[1]*2);
  c->Divide(1,2);
  c->cd(1);
  displayHistEtaPhi_->Draw();
  c->cd(2);
  displayHistEtaPhi_->Draw();
  
  c->ToggleToolBar();
  displayEtaPhi_[0] = c;
  
  c->Draw();
  c->cd();
  DisplayRecHitsEtaPhi();
  c->cd();
  DisplayClustersEtaPhi();


//   // scan for algos, create corresponding display if necessary
//   map<int, bool> isDisplayed;
//   TIter  next(fAllClusters);
//   while(ClusterEF* cluster = (ClusterEF*) next() ) {
//     int algo = cluster->GetType();
    
//     // check that we are supposed to display this algo.
//     set<int>::iterator it0 = fAlgosToDisplay.find(algo);
//     if( it0 == fAlgosToDisplay.end() ) continue; 
    
//     // check that the corresponding canvas has been drawn
//     // if not do it
//     map<int, TCanvas *>::iterator it = fClustersDisplay.find(algo);
//     TCanvas *c = 0;
//     if( it == fClustersDisplay.end() ||
// 	! gROOT->GetListOfCanvases()->FindObject(it->second) ) { 
//       // canvas not found for this algo, create it
//       char cnam[20];
//       sprintf(cnam, "clusters_algo%d", algo);
//       c = new TCanvas(cnam, 
// 		      cnam,
// 		      fViewSize[0], fViewSize[1]*2);
      
//       c->ToggleToolBar();
//       fClustersDisplay[algo] = c;
//       // cout<<"creating new canvas"<<endl;
//       c->Divide(1,2);
//       //RB060206 PrepareClusterCanvas(c, maxeecal, maxehcal);    
//     }
//     else c = it->second;
//     map<int, bool>::iterator itIsDisplayed = isDisplayed.find(algo);
//     if (itIsDisplayed == isDisplayed.end()) {
//       bool isMadeOfSimCells = cluster->IsMadeOfSimCells();
//       PrepareClusterCanvas(c, maxeecal, maxehcal, isMadeOfSimCells);    
//       isDisplayed[algo] = true;
//     }       
    
//     // ATTENTION : clusters from ECAL+HCAL towers with energy in 
//     // both calos are not drawn - as in original code
//     // cout<<"DEBUG zob"<<" "<<cluster<<endl;
//     if( cluster->GetEECAL() && !cluster->GetEHCAL() 
// 	|| cluster->BelongsTo() == LAYER_ECAL_ENDCAP_PRESHOWER_1 
// 	|| cluster->BelongsTo() == LAYER_ECAL_ENDCAP_PRESHOWER_2 ) {
//       c->cd(1);
//       // cout<<"DEBUG ecal"<<endl;
//     }
//     else if( !cluster->GetEECAL() && cluster->GetEHCAL() ) {
//       c->cd(2);
//       // cout<<"DEBUG hcal"<<endl;
//     }
//     else {
//       // cout<<"DEBUG continue"<<endl;
//       continue;
//     }
    
//     // cout<<"DEBUG drawing cluster "<<endl;
//     // cout<<(*cluster)<<endl;
//     cluster->Draw();
//   }  
}


// void EventEF::PrepareClusterCanvas(TCanvas *c, double maxeecal, 
// 				   double maxehcal, bool isSimCells) {

//   if( !gROOT->GetListOfCanvases()->FindObject(c) ) return;

//   if(!displayHistEtaPhi_) {
//     displayHistEtaPhi_ = new TH2F("fHist","",100,-5,5,100,-3.2,3.2);
//     displayHistEtaPhi_->SetXTitle("Eta");
//     displayHistEtaPhi_->SetYTitle("Phi");
//     displayHistEtaPhi_->SetStats(0);
//   }

//   c->cd(1);
//   displayHistEtaPhi_->Draw();
//   c->cd(2);
//   displayHistEtaPhi_->Draw();
  
//   DisplayRecHitsEtaPhi();

//   // display cells
//   double maxe = 0;
//   double thresh=0;
//   TIter* next;
//   if (!isSimCells)
//     next = new TIter(fAllCells);
//   else
//     next = new TIter(fAllSimCells);
//   while(CellEF* cell = (CellEF*)((*next)()) ) {
//     switch( cell->GetLayer() ) {
//     case LAYER_ECAL_BARREL:
//       maxe = maxeecal;
//       thresh = fDisplayECALthresh;
//       c->cd(1);
//       break;
//     case LAYER_ECAL_ENDCAP:
//       maxe = maxeecal;
//       thresh = fDisplayECALecthresh;
//       c->cd(1);
//       break;
//     case LAYER_ECAL_ENDCAP_PRESHOWER_1:
//     case LAYER_ECAL_ENDCAP_PRESHOWER_2:
//       maxe = maxeecal;
//       thresh = fDisplayECALpreshthresh;
//       c->cd(1);
//       break;
//     case LAYER_HCAL_BARREL_1:
//     case LAYER_HCAL_BARREL_2:
//       maxe = maxehcal;
//       thresh = fDisplayHCALthresh;
//       c->cd(2);
//       break;
//     case LAYER_HCAL_ENDCAP_1:
//     case LAYER_HCAL_ENDCAP_2:
//       maxe = maxehcal;
//       thresh = fDisplayHCALecthresh;
//       c->cd(2);
//       break;
//     case LAYER_VFCAL:
//       maxe = maxehcal;
//       thresh = fDisplayVFCALthresh;
//       c->cd(2);
//       break;
//     default:
//       assert(0);
//     }

//     cell->Draw(maxe, thresh);
//   }
//   delete next;
// }

void PFRootEventManager::DisplayView(unsigned viewType) 
{
  // Clear TGraph if needed
  if (graphTrack_[viewType].size()) {
    for (unsigned iGraph = 0; iGraph < graphTrack_[viewType].size(); iGraph++)
      delete graphTrack_[viewType][iGraph];
    graphTrack_[viewType].clear();
  }

  // Display or clear canvas
  if(!displayView_[viewType] || !gROOT->GetListOfCanvases()->FindObject(displayView_[viewType]) ) {
    assert(viewSize_.size() == 2);
    switch(viewType) {
    case XY:
      displayView_[viewType] = new TCanvas("displayXY_", "XY view",
					   viewSize_[0], viewSize_[1]);
      break;
    case RZ:
      displayView_[viewType] = new TCanvas("displayRZ_", "RZ view",
					   viewSize_[0], viewSize_[1]);
      break;
    }
    displayView_[viewType]->SetGrid(0, 0);
    displayView_[viewType]->SetLeftMargin(0.12);
    displayView_[viewType]->SetBottomMargin(0.12);
    displayView_[viewType]->Draw();  
  } else 
    displayView_[viewType]->Clear();
  displayView_[viewType]->cd();

  // Draw support histogram
  double zLow = -500.;
  double zUp  = +500.;
  double rLow = -300.;
  double rUp  = +300.;
  if(!displayHist_[viewType]) {
    switch(viewType) {
    case XY:
      displayHist_[viewType] = new TH2F("hdisplayHist_XY", "", 
					500, rLow, rUp, 500, rLow, rUp);
      displayHist_[viewType]->SetXTitle("X");
      displayHist_[viewType]->SetYTitle("Y");
      break;
    case RZ:
      displayHist_[viewType] = new TH2F("hdisplayHist_RZ", "", 
					500, zLow, zUp, 500, rLow, rUp);
      displayHist_[viewType]->SetXTitle("Z");
      displayHist_[viewType]->SetYTitle("R");
      break;
    default:
      std::cerr << "This kind of view is not implemented" << std::endl;
      break;
    }
    displayHist_[viewType]->SetStats(kFALSE);
  }
  displayHist_[viewType]->Draw();

  if (viewType == XY) {
    // Draw ECAL front face
    frontFaceECALXY_.SetX1(0);
    frontFaceECALXY_.SetY1(0);
    frontFaceECALXY_.SetR1(129); // <==== BE CAREFUL, ECAL size is hardcoded !!!!
    frontFaceECALXY_.SetR2(129);
    frontFaceECALXY_.SetFillStyle(0);
    frontFaceECALXY_.Draw();

    // Draw HCAL front face
    frontFaceHCALXY_.SetX1(0);
    frontFaceHCALXY_.SetY1(0);
    frontFaceHCALXY_.SetR1(183); // <==== BE CAREFUL, HCAL size is hardcoded !!!!
    frontFaceHCALXY_.SetR2(183);
    frontFaceHCALXY_.SetFillStyle(0);
    frontFaceHCALXY_.Draw();
  } else if (viewType == RZ) {
    // Draw lines at different etas
    TLine l;
    l.SetLineColor(1);
    l.SetLineStyle(3);
    TLatex etaLeg;
    etaLeg.SetTextSize(0.02);
    float etaMin = -3.;
    float etaMax = +3.;
    float etaBin = 0.2;
    int nEtas = int((etaMax - etaMin)/0.2) + 1;
    for (int iEta = 0; iEta <= nEtas; iEta++) {
      float eta = etaMin + iEta*etaBin;
      float r = 0.9*rUp;
      TVector3 etaImpact;
      etaImpact.SetPtEtaPhi(r, eta, 0.);
      etaLeg.SetTextAlign(21);
      if (eta <= -1.39) {
	etaImpact.SetXYZ(0.,0.85*zLow*tan(etaImpact.Theta()),0.85*zLow);
	etaLeg.SetTextAlign(31);
      } else if (eta >= 1.39) {
	etaImpact.SetXYZ(0.,0.85*zUp*tan(etaImpact.Theta()),0.85*zUp);
	etaLeg.SetTextAlign(11);
      }
      l.DrawLine(0., 0., etaImpact.Z(), etaImpact.Perp());
      etaLeg.DrawLatex(etaImpact.Z(), etaImpact.Perp(), Form("%2.1f", eta));
    }
    
    frontFaceECALRZ_.SetX1(-303.16);
    frontFaceECALRZ_.SetY1(-129.);
    frontFaceECALRZ_.SetX2(303.16);
    frontFaceECALRZ_.SetY2(129.);
    frontFaceECALRZ_.SetFillStyle(0);
    frontFaceECALRZ_.Draw();
  }

  // Put highest Pt reconstructed track along phi = pi/2
  double phi0 = 0.;
//   double ptMax = 0.;
//   double pxMax = 0.;
//   double pyMax = 0.;
//   std::vector<reco::PFRecTrack>::iterator itRecTrack;
//   for (itRecTrack = recTracks_.begin(); itRecTrack != recTracks_.end();
//        itRecTrack++) {
//     double pt = itRecTrack->trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach).momentum().Pt();
//     if (pt > ptMax) {
//       ptMax = pt;
//       phi0 = itRecTrack->trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach).momentum().Phi();
//       pxMax = itRecTrack->trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach).momentum().Px();
//       pyMax = itRecTrack->trajectoryPoint(reco::PFTrajectoryPoint::ClosestApproach).momentum().Py();
//     }
//   }


  // Display reconstructed objects
  displayView_[viewType]->cd();
  DisplayRecHits(viewType, phi0);
  DisplayRecTracks(viewType, phi0);
  DisplayClusters(viewType, phi0);
}

void PFRootEventManager::DisplayRecHitsEtaPhi() {
  
  TPad *canvas = dynamic_cast<TPad*>( TPad::Pad() ); 
  if(! canvas ) return;

  
  double maxee = GetMaxEEcal();
  double maxeh = GetMaxEHcal();
  
  for( unsigned i=0; i<rechits_.size(); i++) {
    reco::PFRecHit& rh = rechits_[i];
    
    double maxe = 0;
    double thresh = 0;
    switch( rh.GetLayer() ) {
    case PFLayer::ECAL_BARREL:
      maxe = maxee;
      thresh = threshEcalBarrel_;
      canvas->cd(1);
      break;     
    case PFLayer::ECAL_ENDCAP:
      maxe = maxee;
      thresh = threshEcalEndcap_;
      canvas->cd(1);
      break;     
    case PFLayer::HCAL_BARREL1:
    case PFLayer::HCAL_BARREL2:
    case PFLayer::HCAL_ENDCAP:
      maxe = maxeh;
      thresh = 0;
      canvas->cd(2);
      break;     
    default:
      cerr<<"manage other layers"<<endl;
      continue;
    }

    if(rh.GetEnergy() > thresh )
      DisplayRecHitEtaPhi(rh, maxe, thresh);
  }
}


void   PFRootEventManager::DisplayRecHitEtaPhi(reco::PFRecHit& rh, 
					       double maxe, double thresh) {
  
//   TMarker m;
 
//   m.SetMarkerStyle(20);
//   m.DrawMarker( rh.GetPosition().Eta(), 
// 		rh.GetPosition().Phi() );

  
  if ( rh.GetEnergy() < thresh ) return;
  
  double eta = rh.GetPositionREP().Eta();
  double phi = rh.GetPositionREP().Phi();

  // cout<<"display rechit "<<eta<<" "<<phi<<endl;

  // is there a cutg defined ? if yes, draw only if the rechit is inside
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if(cutg) {
    if( !cutg->IsInside(eta, phi) ) return;
  }


  int color = TColor::GetColor(220, 220, 255);
  if(rh.IsSeed() == 1) {
//     color = TColor::GetColor(150, 200, 255);
    color = TColor::GetColor(100, 150, 255);
  }
  
  // preshower rechits are drawn in a different color
  if(rh.GetLayer() == PFLayer::PS1 ||
     rh.GetLayer() == PFLayer::PS2 ) {
    color = 6;
    if(rh.IsSeed() == 1) color = 41;
  }


  TPolyLine linesize;
  TPolyLine lineprop;
  
  double etaSize[4];
  double phiSize[4];
  double x[5];
  double y[5];

  const std::vector< math::XYZPoint >& corners = rh.GetCornersXYZ();
  
  assert(corners.size() == 4);

//   vector<TVector3*> corners; // just for easy manipulation of the corners.
//   corners.push_back( &fCorner1 );
//   corners.push_back( &fCorner2 );
//   corners.push_back( &fCorner3 );
//   corners.push_back( &fCorner4 );
  
  double propfact = 0.95; // so that the cells don't overlap ? 
  for ( unsigned jc=0; jc<4; ++jc ) { 
    phiSize[jc] = phi-corners[jc].Phi();
    etaSize[jc] = eta-corners[jc].Eta();
    if ( phiSize[jc] > 1. ) phiSize[jc] -= 2.*TMath::Pi();  // this is strange...
    if ( phiSize[jc] < -1. ) phiSize[jc]+= 2.*TMath::Pi();
    phiSize[jc] *= propfact;
    etaSize[jc] *= propfact;
  }
  
  for ( unsigned jc=0; jc<4; ++jc ) { 
    x[jc] = eta + etaSize[jc];
    y[jc] = phi + phiSize[jc];

    // if( fLayer==-1 ) cout<<"DEBUG "<<jc<<" "<<x<<" "<<y<<" "<<etaSize[jc]<<" "<<phiSize[jc]<<endl;
  }
  x[4]=x[0];
  y[4]=y[0]; // closing the polycell

  linesize.SetLineColor(color);
  linesize.DrawPolyLine(5, x, y);

  // we do not draw area prop to energy for preshower rechits
  if(rh.GetLayer() == PFLayer::PS1 || 
     rh.GetLayer() == PFLayer::PS2) 
    return;

  // in addition,
  // log weighting of the polyline sides w/r to the max energy 
  // in the detector
  double etaAmpl = (log(rh.GetEnergy()+1.)/log(maxe+1.));
  double phiAmpl = (log(rh.GetEnergy()+1.)/log(maxe+1.));
  for ( unsigned jc=0; jc<4; ++jc ) { 
    x[jc] = eta + etaSize[jc]*etaAmpl;
    y[jc] = phi + phiSize[jc]*phiAmpl;
  }
  x[4]=x[0];
  y[4]=y[0];
  
  
  lineprop.SetLineColor(color);
  lineprop.SetFillColor(color);
  lineprop.DrawPolyLine(5,x,y,"f");

//   if(fDrawID) {
//     char id[10];
//     sprintf(id, "%d", fID);
//     TLatex lid( eta, phi, id);
//     lid.DrawLatex(eta, phi, id);
//   }
//   return;
}


void PFRootEventManager::DisplayClustersEtaPhi() {

  TPad *canvas = dynamic_cast<TPad*>( TPad::Pad() ); 
  if(! canvas ) return;

  
  for( unsigned i=0; i<clusters_.size(); i++) {
    // cout<<"displaying "<<clusters_[i]<<" "<<clusters_[i].GetLayer()<<endl;
    
    if( clusters_[i].GetLayer() < 0 ) { // ECAL
      canvas->cd(1);
      DisplayClusterEtaPhi(clusters_[i]);
    } 
    else { // HCAL 
      canvas->cd(2);
      DisplayClusterEtaPhi(clusters_[i]);
    }
  }
}

void PFRootEventManager::DisplayClusterEtaPhi(const reco::PFCluster& cluster) {
  TMarker m;
  m.SetMarkerColor(4);
  m.SetMarkerStyle(20);

  double eta = cluster.GetPositionREP().Eta();
  double phi = cluster.GetPositionREP().Phi();
  
  m.DrawMarker(eta, phi);
}

void PFRootEventManager::DisplayRecHits(unsigned viewType, double phi0) 
{
  double maxee = GetMaxEEcal();
  double maxeh = GetMaxEHcal();
  
  std::vector<reco::PFRecHit>::iterator itRecHit;
  for (itRecHit = rechits_.begin(); itRecHit != rechits_.end(); 
       itRecHit++) {
    double maxe = 0;
    double thresh = 0;
    switch(itRecHit->GetLayer()) {
    case PFLayer::ECAL_BARREL:
      maxe = maxee;
      thresh = threshEcalBarrel_;
      break;     
    case PFLayer::ECAL_ENDCAP:
      maxe = maxee;
      thresh = threshEcalEndcap_;
      break;     
    case PFLayer::HCAL_BARREL1:
    case PFLayer::HCAL_BARREL2:
      maxe = maxeh;
      thresh = threshHcalBarrel_;
      break;           
    case PFLayer::HCAL_ENDCAP:
      maxe = maxeh;
      thresh = threshHcalEndcap_;
      break;           
    default:
      cerr<<"manage other layers"<<endl;
      continue;
    }

    if(itRecHit->GetEnergy() > thresh )
      DisplayRecHit(*itRecHit, viewType, maxe, thresh, phi0);
  }
}

void PFRootEventManager::DisplayRecHit(reco::PFRecHit& rh, unsigned viewType,
				       double maxe, double thresh,
				       double phi0) 
{
  math::XYZPoint vPhi0(cos(phi0), sin(phi0), 0.);
  if (rh.GetEnergy() < thresh ) return;

  double eta = rh.GetPositionXYZ().Eta();
  double phi = rh.GetPositionXYZ().Phi();
  double sign = 1.;
  if (cos(phi0 - phi) < 0.) sign = -1.;

  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if (cutg) {
    if( !cutg->IsInside(eta, phi) ) return;
  }

  double etaSize[4];
  double phiSize[4];
  double x[5];
  double y[5];
  double z[5];
  double r[5];
  double xprop[5];
  double yprop[5];
  
  const std::vector< math::XYZPoint >& corners = rh.GetCornersXYZ();
  
  assert(corners.size() == 4);
  
  double propfact = 0.95; // so that the cells don't overlap ? 
  double ampl = (log(rh.GetEnergy() + 1.)/log(maxe + 1.));
  
  for ( unsigned jc=0; jc<4; ++jc ) { 
    phiSize[jc] = phi-corners[jc].Phi();
    etaSize[jc] = eta-corners[jc].Eta();
    if ( phiSize[jc] > 1. ) phiSize[jc] -= 2.*TMath::Pi();  // this is strange...
    if ( phiSize[jc] < -1. ) phiSize[jc]+= 2.*TMath::Pi();
    phiSize[jc] *= propfact;
    etaSize[jc] *= propfact;

    math::XYZPoint cornerposxyz(corners[jc].X()*vPhi0.Y() - corners[jc].Y()*vPhi0.X(), corners[jc].X()*vPhi0.X() + corners[jc].Y()*vPhi0.Y(), corners[jc].Z());
    // cornerposxyz.SetPhi( corners[jc]->Y() - phi0 );

    x[jc] = cornerposxyz.X();
    y[jc] = cornerposxyz.Y();
    z[jc] = cornerposxyz.Z();
    r[jc] = sign*cornerposxyz.Rho();

    // cell area is prop to log(E)
    if(!(rh.GetLayer() == PFLayer::ECAL_BARREL || 
	 rh.GetLayer() == PFLayer::HCAL_BARREL1 || 
	 rh.GetLayer() == PFLayer::HCAL_BARREL2)) {
      
      math::XYZPoint centreXYZrot(rh.GetPositionXYZ().X()*vPhi0.Y() - rh.GetPositionXYZ().Y()*vPhi0.X(), rh.GetPositionXYZ().X()*vPhi0.X() + rh.GetPositionXYZ().Y()*vPhi0.Y(), rh.GetPositionXYZ().Z());
      // centreXYZrot.SetPhi( fCentre.Y() - phi0 );

      math::XYZPoint centertocorner(x[jc] - centreXYZrot.X(), 
				    y[jc] - centreXYZrot.Y(), 0.);
      // centertocorner -= centreXYZrot;
      xprop[jc] = centreXYZrot.X() + centertocorner.X()*ampl;
      yprop[jc] = centreXYZrot.Y() + centertocorner.Y()*ampl;
    }
  }

  if(rh.GetLayer() == PFLayer::ECAL_BARREL || 
     rh.GetLayer() == PFLayer::HCAL_BARREL1 || 
     rh.GetLayer() == PFLayer::HCAL_BARREL2 || viewType == RZ) {

    // we are in the barrel. Determining which corners to shift 
    // away from the center to represent the cell energy
    int i1 = -1;
    int i2 = -1;

    if(fabs(phiSize[1]-phiSize[0]) > 0.0001) {
      if (viewType == XY) {
	i1 = 2;
	i2 = 3;
      } else if (viewType == RZ) {
	i1 = 1;
	i2 = 2;
      }
    } else {
      if (viewType == XY) {
	i1 = 1;
	i2 = 2;
      } else if (viewType == RZ) {
	i1 = 2;
	i2 = 3;
      }
    }

    x[i1] *= 1+ampl/2.;
    x[i2] *= 1+ampl/2.;
    y[i1] *= 1+ampl/2.;
    y[i2] *= 1+ampl/2.;
    z[i1] *= 1+ampl/2.;
    z[i2] *= 1+ampl/2.;
    r[i1] *= 1+ampl/2.;
    r[i2] *= 1+ampl/2.;
  }

  
  x[4]=x[0];
  y[4]=y[0]; // closing the polycell
  z[4]=z[0];
  r[4]=r[0]; // closing the polycell
  

  int color = TColor::GetColor(220, 220, 255);
  if(rh.IsSeed() == 1) {
    color = TColor::GetColor(100, 150, 255);
  }

  if (viewType == XY) {
    TPolyLine lineSizeXY;
    TPolyLine linePropXY;          
    if(rh.GetLayer() == PFLayer::ECAL_BARREL || 
       rh.GetLayer() == PFLayer::HCAL_BARREL1 || 
       rh.GetLayer() == PFLayer::HCAL_BARREL2) {
      lineSizeXY.SetLineColor(color);
      //cout << "x,y " << x[0] << " " << y[0] << endl;
      lineSizeXY.SetFillColor(color);
      lineSizeXY.DrawPolyLine(5,x,y,"f");
    } else {
      //cout << "x,y " << x[0] << " " << y[0] << endl;
      lineSizeXY.SetLineColor(color);
      lineSizeXY.DrawPolyLine(5,x,y);
      
      xprop[4]=xprop[0];
      yprop[4]=yprop[0]; // closing the polycell    
      linePropXY.SetLineColor(color);
      linePropXY.SetFillColor(color);
      linePropXY.DrawPolyLine(5,xprop,yprop,"F");
    }
  } else if (viewType == RZ) {
    TPolyLine lineSizeRZ;
    lineSizeRZ.SetLineColor(color);
    lineSizeRZ.SetFillColor(color);
    // cout << "z,r " << z[0] << " " << r[0] << endl;
    lineSizeRZ.DrawPolyLine(5,z,r,"f");
  }
}

void PFRootEventManager::DisplayClusters(unsigned viewType, double phi0)
{
  math::XYZPoint vPhi0(cos(phi0), sin(phi0), 0.);
  std::vector<reco::PFCluster>::iterator itCluster;
  for (itCluster = clusters_.begin(); itCluster != clusters_.end(); 
       itCluster++) {
    TMarker m;
    m.SetMarkerColor(4);
    m.SetMarkerStyle(20);

    math::XYZPoint xyzPos(itCluster->GetPositionXYZ().X()*vPhi0.Y() - itCluster->GetPositionXYZ().Y()*vPhi0.X(), itCluster->GetPositionXYZ().X()*vPhi0.X() + itCluster->GetPositionXYZ().Y()*vPhi0.Y(), itCluster->GetPositionXYZ().Z());

    switch(viewType) {
    case XY:
      m.DrawMarker(xyzPos.X(), xyzPos.Y());
      break;
    case RZ:
      double sign = 1.;
      if (cos(phi0 - itCluster->GetPositionXYZ().Phi()) < 0.)
	sign = -1.;
      m.DrawMarker(xyzPos.Z(), sign*xyzPos.Rho());
      break;
    }      
  }
}

void PFRootEventManager::DisplayRecTracks(unsigned viewType, double phi0) 
{
  math::XYZPoint vPhi0(cos(phi0), sin(phi0), 0.);
  std::vector<reco::PFRecTrack>::iterator itRecTrack;
  for (itRecTrack = recTracks_.begin(); itRecTrack != recTracks_.end();
       itRecTrack++) {
    double sign = 1.;
    if (cos(phi0 - itRecTrack->trajectoryPoint(itRecTrack->nTrajectoryMeasurements() + reco::PFTrajectoryPoint::ECALEntrance).momentum().Phi()) < 0.)
      sign = -1.;

    // Check number of measurements with non-zero momentum
    std::vector<reco::PFTrajectoryPoint> trajectoryPoints = 
      itRecTrack->trajectoryPoints();
    std::vector<reco::PFTrajectoryPoint>::iterator itTrajPt;
    unsigned nValidPts = 0;
    for (itTrajPt = trajectoryPoints.begin(); 
	 itTrajPt != trajectoryPoints.end(); itTrajPt++)
      if (itTrajPt->momentum().P() > 0.) nValidPts++;
    if (!nValidPts) continue;
    double* xPos = new double[nValidPts];
    double* yPos = new double[nValidPts];
    unsigned iValid = 0;
    // cout << "Draw a new track " << nValidPts << endl;
    for (itTrajPt = trajectoryPoints.begin(); 
	 itTrajPt != trajectoryPoints.end(); itTrajPt++)
      if (itTrajPt->momentum().P() > 0.) {
	math::XYZPoint xyzPos(itTrajPt->xyzPosition().X()*vPhi0.Y() - itTrajPt->xyzPosition().Y()*vPhi0.X(), itTrajPt->xyzPosition().X()*vPhi0.X() + itTrajPt->xyzPosition().Y()*vPhi0.Y(), itTrajPt->xyzPosition().Z());
	// xyzPos.SetPhi(xyzPos.Phi()); // <=== Does not work ??? why ???
	switch(viewType) {
	case XY:
	  xPos[iValid] = xyzPos.X();
	  yPos[iValid] = xyzPos.Y();
	  // cout << "\t" << itTrajPt->xyzPosition().X() << " " 
	  //     << itTrajPt->xyzPosition().Y() << endl;
	  break;
	case RZ:
	  xPos[iValid] = xyzPos.Z();
	  yPos[iValid] = sign*xyzPos.Rho();
	  break;
	}
	iValid++;
      }
    graphTrack_[viewType].push_back(new TGraph(nValidPts, xPos, yPos));
    int color = 103;
    unsigned lastHisto = graphTrack_[viewType].size() - 1;
    graphTrack_[viewType][lastHisto]->SetMarkerColor(color);
    graphTrack_[viewType][lastHisto]->SetMarkerStyle(8);
    graphTrack_[viewType][lastHisto]->SetMarkerSize(0.5);
    graphTrack_[viewType][lastHisto]->SetLineColor(color);
    graphTrack_[viewType][lastHisto]->SetLineStyle(itRecTrack->algoType());
    graphTrack_[viewType][lastHisto]->Draw("pl");
    delete[] xPos;
    delete[] yPos;
  }
  return;
}

double PFRootEventManager::GetMaxEEcal() {
  
  if( maxERecHitEcal_<0 ) {
    double maxeec = GetMaxE( PFLayer::ECAL_ENDCAP );
    double maxeb =  GetMaxE( PFLayer::ECAL_BARREL );
    maxERecHitEcal_ = maxeec > maxeb ? maxeec:maxeb; 
    // max of both barrel and endcap
  }
  return  maxERecHitEcal_;
}




double PFRootEventManager::GetMaxEHcal() {

  if(maxERecHitHcal_ < 0) {
    double maxeec = GetMaxE( PFLayer::HCAL_ENDCAP );
    double maxeb =  GetMaxE( PFLayer::HCAL_BARREL1 );
    maxERecHitHcal_ =  maxeec>maxeb  ?  maxeec:maxeb;
  }
  return maxERecHitHcal_;
}


double PFRootEventManager::GetMaxE(int layer) const {

  double maxe = -9999;

  for( unsigned i=0; i<rechits_.size(); i++) {
    if( rechits_[i].GetLayer() != layer ) continue;
    if( rechits_[i].GetEnergy() > maxe)
      maxe = rechits_[i].GetEnergy();
  }

  return maxe;
}

