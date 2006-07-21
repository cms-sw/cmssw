#include "DataFormats/PFReco/interface/PFLayer.h"
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

#include <iostream>

using namespace std;

PFRootEventManager::PFRootEventManager() {}

PFRootEventManager::~PFRootEventManager() {}

PFRootEventManager::PFRootEventManager(const char* file) {  
  options_ = 0;
  ReadOptions(file);
  iEvent_=0;
  displayHistEtaPhi_=0;
  maxERecHitEcal_ = -1;
  maxERecHitHcal_ = -1;
  
}

void PFRootEventManager::Reset() { 
  maxERecHitEcal_ = -1;
  maxERecHitHcal_ = -1;  
  clusters_.clear();
  rechits_.clear();
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
    return;
  }

  hitsBranch_->SetAddress(&rechits_);


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
    cerr<<"EventEF::ReadOptions, bad display/viewsize_etaphi tag...using 700/350"
	<<endl;
    viewSizeEtaPhi_.clear();
    viewSizeEtaPhi_.push_back(700); 
    viewSizeEtaPhi_.push_back(350); 
  }

  viewSizeXY_.clear();
  options_->GetOpt("display", "viewsize_etaphi", viewSizeXY_);
  if(viewSizeXY_.size() != 2) {
    cerr<<"EventEF::ReadOptions, bad display/viewsize_xy tag...using 700/350"
	<<endl;
    viewSizeXY_.clear();
    viewSizeXY_.push_back(600); 
    viewSizeXY_.push_back(600); 
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


  threshPS_ = 0.0001;
  options_->GetOpt("clustering", "thresh_PS", threshPS_);
  
  threshSeedPS_ = 0.001;
  options_->GetOpt("clustering", "thresh_Seed_PS", 
		   threshSeedPS_);


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

  cout<<"process entry"<<endl;
  
  hitsBranch_->GetEntry(entry);

  cout<<"numer of rechits : "<<rechits_.size()<<endl;
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

  clusteralgo.Init( rechits ); 
  clusteralgo.AllClusters();

  clusters_ = clusteralgo.GetClusters();
}


void PFRootEventManager::Display(int ientry) {
  
  ProcessEntry(ientry);
  DisplayEtaPhi();
  DisplayXY();
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
    case PFLayer::PS1:
    case PFLayer::PS2:
      maxe = -1;
      thresh = threshPS_;
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
  bool isseed = false;
  if(isseed) {
    color = TColor::GetColor(150, 200, 255);
  }
  
  // preshower rechits are drawn in a different color
  if(rh.GetLayer() == PFLayer::PS1 ||
     rh.GetLayer() == PFLayer::PS2 ) {
    color = 6;
    if(isseed) color = 41;
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
//     TMarker m;
//     m.SetMarkerColor(4);
//     m.SetMarkerStyle(20);
    
//     double xm = corners[jc].Eta();
//     double ym = corners[jc].Phi();
    
//     m.DrawMarker(xm,ym);


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

  if(maxe>0) {
    
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
  }

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
    cout<<"displaying "<<clusters_[i]<<" "<<clusters_[i].GetLayer()<<endl;
    
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


void PFRootEventManager::DisplayXY() {

  if(!displayHistXY_) {
    displayHistXY_ = new TH2F("fHistXY","",100,-300,300,100,-300,300);
    displayHistXY_->SetXTitle("X");
    displayHistXY_->SetYTitle("Y");
    displayHistXY_->SetStats(0);
  }

  if(displayXY_) 
    displayXY_->Clear();
  else 
    displayXY_ = new TCanvas("xy", 
			     "X/Y view",
			     viewSizeXY_[0], viewSizeXY_[1]);



  displayHistXY_->Draw();

  TPad *canvas = dynamic_cast<TPad*>( TPad::Pad() ); 
  if(! canvas ) return;

  double maxee = GetMaxEEcal();
  double maxeh = GetMaxEHcal();
  
  for( unsigned i=0; i<rechits_.size(); i++) {
    reco::PFRecHit& rh = rechits_[i];
    
    double maxe = -1;
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
    case PFLayer::PS1:
    case PFLayer::PS2:
      maxe = -1;
      thresh = threshPS_;
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
    DisplayRecHitXY(rh, maxe, thresh);
  }
}


void PFRootEventManager::DisplayRecHitXY(reco::PFRecHit& rh,
					 double maxe, double thresh) {
  
  if ( rh.GetEnergy() < thresh ) return;
  
  double xpos = rh.GetPositionXYZ().X();
  double ypos = rh.GetPositionXYZ().Y();
  
  cout<<"display rechit "<<xpos<<" "<<ypos<<endl;

  // is there a cutg defined ? if yes, draw only if the rechit is inside
  TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  if(cutg) {
    if( !cutg->IsInside(xpos, ypos) ) return;
  }

  int color = TColor::GetColor(220, 220, 255);
  bool isseed = false;
  if(isseed) {
    color = TColor::GetColor(150, 200, 255);
  }
  
  // preshower rechits are drawn in a different color
  if(rh.GetLayer() == PFLayer::PS1 ||
     rh.GetLayer() == PFLayer::PS2 ) {
    color = 6;
    if(isseed) color = 41;
  }


  TPolyLine linesize;
  TPolyLine lineprop;
  
  double xSize[4];
  double ySize[4];
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
//     cout<<"  corner : "<<corners[jc].X()<<" "<<corners[jc].Y()<<endl;

//     TMarker m;
//     m.SetMarkerColor(4);
//     m.SetMarkerStyle(20);
    
//     double xm = corners[jc].X();
//     double ym = corners[jc].Y();
  
//     m.DrawMarker(xm,ym);


    ySize[jc] = ypos-corners[jc].Y();
    xSize[jc] = xpos-corners[jc].X();
    if ( ySize[jc] > 1. ) ySize[jc] -= 2.*TMath::Pi();  // this is strange...
    if ( ySize[jc] < -1. ) ySize[jc]+= 2.*TMath::Pi();
    ySize[jc] *= propfact;
    xSize[jc] *= propfact;
  }
  
  for ( unsigned jc=0; jc<4; ++jc ) { 
    x[jc] = xpos + xSize[jc];
    y[jc] = ypos + ySize[jc];

    // if( fLayer==-1 ) cout<<"DEBUG "<<jc<<" "<<x<<" "<<y<<" "<<x Size[jc]<<" "<<ySize[jc]<<endl;
  }
  x[4]=x[0];
  y[4]=y[0]; // closing the polycell

  linesize.SetLineColor(color);
  linesize.DrawPolyLine(5, x, y);

  // we do not draw area prop to energy for preshower rechits
  if(rh.GetLayer() == PFLayer::PS1 || 
     rh.GetLayer() == PFLayer::PS2) 
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

