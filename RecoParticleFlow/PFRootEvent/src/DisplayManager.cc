#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "RecoParticleFlow/PFRootEvent/interface/IO.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "RecoParticleFlow/PFTracking/interface/PFGeometry.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/Math/interface/Point3D.h"

#include "DataFormats/FWLite/interface/ChainEvent.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFRecHit.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFTrack.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFSimParticle.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFGenParticle.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h"

#include <TH2.h>
#include <TTree.h>
#include <TVector3.h>
#include <TH2F.h>
#include <TEllipse.h>
#include <TLine.h>
#include <TLatex.h>
#include <TList.h>
#include <TColor.h>
#include <TMath.h>
#include <TApplication.h>

using namespace std;

//________________________________________________________________

DisplayManager::DisplayManager(PFRootEventManager *em,
                               const char* optfile ) : 
  em_(em), 
  options_(0),
  maxERecHitEcal_(-1),
  maxERecHitHcal_(-1),
  maxERecHitHo_(-1),
  isGraphicLoaded_(false),
  shiftId_(SHIFTID) {
        
  readOptions( optfile );
        
  eventNumber_  = em_->eventNumber();
  //TODOLIST: re_initialize if new option file- better in em
  maxEvents_= em_->ev_->size();  
  createCanvas();
}
//________________________________________________________
DisplayManager::~DisplayManager()
{
  reset();
} 

//________________________________________________________
void DisplayManager::readOptions( const char* optfile ) {
        
  try {
    delete options_;
    options_ = new IO(optfile);
  }
  catch( const string& err ) {
    cout<<err<<endl;
    return;
  }
      
  drawHO_=true;
  options_->GetOpt("display", "drawHO", drawHO_);  



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
        
  clusterAttributes_.clear();
  options_->GetOpt("display", "cluster_attributes", clusterAttributes_);
  if(clusterAttributes_.size() != 7) {
    cerr<<"PFRootEventManager::::ReadOptions, bad display/cluster_attributes tag...using 20 10 2 5"
        <<endl;
    clusterAttributes_.clear();
    clusterAttributes_.push_back(2); //color
    clusterAttributes_.push_back(5);  // color if clusterPS
    clusterAttributes_.push_back(20); // marker style
    clusterAttributes_.push_back(1.0); 
    clusterAttributes_.push_back(6); //For ECAL
    clusterAttributes_.push_back(9); //For HF_EM
    clusterAttributes_.push_back(46); //For HO
  }

  trackAttributes_.clear();
  options_->GetOpt("display", "track_attributes", trackAttributes_);
  if(trackAttributes_.size() != 4) {
    cerr<<"PFRootEventManager::::ReadOptions, bad display/track_attributes tag...using 103 1 8 8"
        <<endl;
    trackAttributes_.clear();
    trackAttributes_.push_back(103); //color Line and Marker
    trackAttributes_.push_back(1);  //line style
    trackAttributes_.push_back(8);   //Marker style
    trackAttributes_.push_back(1.0);   //Marker size
  }
  gsfAttributes_.clear();
  options_->GetOpt("display", "gsf_attributes", gsfAttributes_);
  if(gsfAttributes_.size() != 4) {
    cerr<<"PFRootEventManager::::ReadOptions, bad display/gsf_attributes tag...using 105 1 8 8"
	<<endl;
    gsfAttributes_.clear();
    gsfAttributes_.push_back(105); //color Line and Marker
    gsfAttributes_.push_back(1);  //line style
    gsfAttributes_.push_back(8);   //Marker style
    gsfAttributes_.push_back(1.0);   //Marker size
  }
  bremAttributes_.clear();
  options_->GetOpt("display", "brem_attributes", bremAttributes_);
  if(bremAttributes_.size() != 4) {
    cerr<<"PFRootEventManager::::ReadOptions, bad display/gsf_attributes tag...using 106 1 8 8"
	<<endl;
    bremAttributes_.clear();
    bremAttributes_.push_back(106); //color Line and Marker
    bremAttributes_.push_back(1);  //line style
    bremAttributes_.push_back(8);   //Marker style
    bremAttributes_.push_back(1.0);   //Marker size
  }     
        
  double attrScale = (drawHO_) ? 0.8 : 1.0;

  clusPattern_ = new TAttMarker( (int)clusterAttributes_[0],
				 (int)clusterAttributes_[2],
				 attrScale*clusterAttributes_[3]);

  clusPatternecal_ = new TAttMarker( (int)clusterAttributes_[4],
				 (int)clusterAttributes_[2],
				 attrScale*clusterAttributes_[3]);

  clusPatternhfem_ = new TAttMarker( (int)clusterAttributes_[5],
				 (int)clusterAttributes_[2],
				 attrScale*clusterAttributes_[3]);

  clusPatternho_ = new TAttMarker( (int)clusterAttributes_[6],
				 (int)clusterAttributes_[2],
				 attrScale*clusterAttributes_[3]);

  clusPSPattern_ = new TAttMarker( (int)clusterAttributes_[1],
				   (int)clusterAttributes_[2],
				   attrScale*clusterAttributes_[3]);
  trackPatternL_ = new TAttLine( (int)trackAttributes_[0],
				 (int)trackAttributes_[1],
				 1);
  trackPatternM_ = new TAttMarker( (int)trackAttributes_[0],
				   (int)trackAttributes_[2],
				   attrScale*trackAttributes_[3]);

  gsfPatternL_ = new TAttLine( (int)gsfAttributes_[0],
   			 (int)gsfAttributes_[1],
   			 1);
  gsfPatternM_ = new TAttMarker( (int)gsfAttributes_[0],
				   (int)gsfAttributes_[2],
				   attrScale*gsfAttributes_[3]);

  bremPatternL_ = new TAttLine( (int)bremAttributes_[0],
   			 (int)bremAttributes_[1],
   			 1);
  bremPatternM_ = new TAttMarker( (int)bremAttributes_[0],
				   (int)bremAttributes_[2],
				   attrScale*bremAttributes_[3]);
  
  genPartPattern_= new TAttMarker(kGreen-1,22,1.);
  

  std::vector<float> simPartAttributes;

  simPartAttributes.clear();
  options_->GetOpt("display", "simPart_attributes", simPartAttributes);
  if(simPartAttributes.size() != 3) {
    cerr<<"PFRootEventManager::::ReadOptions, bad display/simPart_attributes tag...using 103 1 8 8"
        <<endl;
    simPartAttributes.clear();
    simPartAttributes.push_back(3); //color Line and Marker
    simPartAttributes.push_back(2);  //line style
    simPartAttributes.push_back(0.6);   
  }
 
  int simColor = (int)simPartAttributes[0];
  int simLStyle = (int)simPartAttributes[1];
  float simMSize = attrScale*simPartAttributes[2];

  simPartPatternPhoton_ = new TAttMarker(simColor,3,simMSize);
  simPartPatternElec_   = new TAttMarker(simColor,5,simMSize);
  simPartPatternMuon_   = new TAttMarker(simColor,2,simMSize);
  simPartPatternK_      = new TAttMarker(simColor,24,simMSize);
  simPartPatternPi_     = new TAttMarker(simColor,25,simMSize);
  simPartPatternProton_ = new TAttMarker(simColor,26,simMSize);
  simPartPatternNeutron_= new TAttMarker(simColor,27,simMSize);
  simPartPatternDefault_= new TAttMarker(simColor,30,simMSize);
        
  simPartPatternL_ = new TAttLine(simColor,simLStyle,1);
  simPartPatternM_.resize(8);
        
  setNewAttrToSimParticles();
        
  drawHits_= true;
  options_->GetOpt("display", "rechits",drawHits_);
        
  drawClus_ = true;
  options_->GetOpt("display", "clusters",drawClus_);
        
  drawClusterL_ = false;
  options_->GetOpt("display", "cluster_lines", drawClusterL_);
        
  drawTracks_ = true;
  options_->GetOpt("display", "rectracks", drawTracks_);

  drawGsfTracks_ = true;
  options_->GetOpt("display", "gsftracks", drawGsfTracks_);

  drawBrems_ = false;
  options_->GetOpt("display", "brems", drawBrems_);
        
  drawParticles_ = true;
  options_->GetOpt("display", "particles", drawParticles_);
        
  particlePtMin_ = -1;
  options_->GetOpt("display", "particles_ptmin", particlePtMin_);
  
  
  drawGenParticles_=false;
  genParticlePtMin_ = 0;
  
  
  trackPtMin_ = -1;
  options_->GetOpt("display", "rectracks_ptmin", trackPtMin_);

  gsfPtMin_ = -1;
  options_->GetOpt("display", "gsfrectracks_ptmin", gsfPtMin_);
        
  hitEnMin_ = -1;
  options_->GetOpt("display","rechits_enmin",hitEnMin_);
        
  clusEnMin_ = -1;
  options_->GetOpt("display","clusters_enmin",clusEnMin_);
        
        
  drawPFBlocks_  = false;
  options_->GetOpt("display","drawPFBlock",drawPFBlocks_);
  //redrawWithoutHits_=false;
        
  zoomFactor_ = 10;  
  options_->GetOpt("display", "zoom_factor", zoomFactor_);

}

//________________________________________________________
void DisplayManager::createCanvas()
{
        
  //TODOLIST: better TCanvas *displayView_[4]
  displayView_.resize(NViews);
  displayHist_.resize(NViews);
        
  // TODOLIST:Canvases size  
  // Add menu to mofify canvas size
  // Add menu of views to be drawn
        
  displayView_[XY] = new TCanvas("displayXY_", "XY view",viewSize_[0], viewSize_[1]);
  displayView_[RZ] = new TCanvas("displayRZ_", "RZ view",viewSize_[0], viewSize_[1]);
  displayView_[EPE] = new TCanvas("displayEPE_", "eta/phi view, ECAL",viewSize_[0], viewSize_[1]);
  displayView_[EPH] = new TCanvas("displayEPH_", "eta/phi view, HCAL",viewSize_[0], viewSize_[1]);
  if (drawHO_) displayView_[EHO] = new TCanvas("displayEHO_", "eta/phi view, HO",viewSize_[0], viewSize_[1]);

  for (int viewType=0;viewType<NViews;++viewType) {
    if (!drawHO_ && viewType==EHO) continue;
    displayView_[viewType]->SetGrid(0, 0);
    displayView_[viewType]->SetBottomMargin(0.14);
    displayView_[viewType]->SetLeftMargin(0.15);
    displayView_[viewType]->SetRightMargin(0.05);
    displayView_[viewType]->ToggleToolBar();
  } 
  // Draw support histogram
  double zLow = -650.;
  double zUp  = +650.;
  double rLow = -450.;
  double rUp  = +450.;
  
  if (!drawHO_) {zLow = -500.; zUp  = +500.; rLow = -300.; rUp  = +300.;}

  displayHist_[XY] = new TH2F("hdisplayHist_XY", "", 500, rLow, rUp, 
                              500, rLow, rUp);
  displayHist_[XY]->SetXTitle("X [cm]");
  displayHist_[XY]->SetYTitle("Y [cm]");
        
  displayHist_[RZ] = new TH2F("hdisplayHist_RZ", "",500, zLow, zUp, 
                              500, rLow, rUp); 
  displayHist_[RZ]->SetXTitle("Z [cm]");
  displayHist_[RZ]->SetYTitle("R [cm]");
        
  displayHist_[EPE] = new TH2F("hdisplayHist_EP", "", 500, -5, 5, 
                               500, -3.5, 3.5);
  displayHist_[EPE]->SetXTitle("#eta");
  displayHist_[EPE]->SetYTitle("#phi [rad]");
        
  displayHist_[EPH] = displayHist_[EPE];

  //  displayHist_[EHO] = new TH2F("hdisplayHist_HO", "", 150, -1.5, 1.5, 
  //  500, -3.5, 3.5);
  //  displayHist_[EHO]->SetXTitle("#eta");
  //  displayHist_[EHO]->SetYTitle("#phi [rad]");
  
  if (drawHO_) displayHist_[EHO] = displayHist_[EPE];
        
  for (int viewType=0;viewType<NViews;++viewType){
    if (!drawHO_ && viewType==EHO) continue;
    displayHist_[viewType]->SetStats(kFALSE);
    displayHist_[viewType]->GetYaxis()->SetTitleSize(0.06);
    displayHist_[viewType]->GetYaxis()->SetTitleOffset(1.2);
    displayHist_[viewType]->GetXaxis()->SetTitleSize(0.06);
    displayHist_[viewType]->GetYaxis()->SetLabelSize(0.045);
    displayHist_[viewType]->GetXaxis()->SetLabelSize(0.045);
  }  
  
  // Draw ECAL front face
  frontFaceECALXY_.SetX1(0);
  frontFaceECALXY_.SetY1(0);
  frontFaceECALXY_.SetR1(PFGeometry::innerRadius(PFGeometry::ECALBarrel));
  frontFaceECALXY_.SetR2(PFGeometry::innerRadius(PFGeometry::ECALBarrel));
  frontFaceECALXY_.SetFillStyle(0);

  // Draw HCAL front face
  frontFaceHCALXY_.SetX1(0);
  frontFaceHCALXY_.SetY1(0);
  frontFaceHCALXY_.SetR1(PFGeometry::innerRadius(PFGeometry::HCALBarrel));
  frontFaceHCALXY_.SetR2(PFGeometry::innerRadius(PFGeometry::HCALBarrel));
  frontFaceHCALXY_.SetFillStyle(0);

  if (drawHO_) {
    // Draw HO front face
    frontFaceHOXY_.SetX1(0);
    frontFaceHOXY_.SetY1(0);
    frontFaceHOXY_.SetR1(PFGeometry::innerRadius(PFGeometry::HOBarrel));
    frontFaceHOXY_.SetR2(PFGeometry::outerRadius(PFGeometry::HOBarrel));
    frontFaceHOXY_.SetFillStyle(0);
  }

  // Draw ECAL side
  frontFaceECALRZ_.SetX1(-1.*PFGeometry::innerZ(PFGeometry::ECALEndcap));
  frontFaceECALRZ_.SetY1(-1.*PFGeometry::innerRadius(PFGeometry::ECALBarrel));
  frontFaceECALRZ_.SetX2(PFGeometry::innerZ(PFGeometry::ECALEndcap));
  frontFaceECALRZ_.SetY2(PFGeometry::innerRadius(PFGeometry::ECALBarrel));
  frontFaceECALRZ_.SetFillStyle(0);
  cout <<"End of DisplayManager::createCanvas()"<<endl; 

}
//_________________________________________________________________________
void DisplayManager::createGCluster(const reco::PFCluster& cluster, 
                                    int ident, 
                                    double phi0)
{

  double eta = cluster.position().Eta();
  double phi = cluster.position().Phi();
  
  //   int type = cluster.type();
  //   if(algosToDisplay_.find(type) == algosToDisplay_.end() )
  //     return;
        
  //   TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");  
  //   if( cutg && !cutg->IsInside( eta, phi ) ) return;
        
        
  //int color = clusterAttributes_[2];
  //if ( cluster.layer()==PFLayer::PS1 || cluster.layer()==PFLayer::PS2 )
  //  color = clusterAttributes_[3];
    
  //int markerSize = clusterAttributes_[1];
  //int markerStyle = clusterAttributes_[0];
        
  int clusType=0;
        
  if ( cluster.layer()==PFLayer::PS1 || cluster.layer()==PFLayer::PS2 )
    clusType=1;  
  
  if ( cluster.layer()==PFLayer::ECAL_BARREL || cluster.layer()==PFLayer::ECAL_ENDCAP)
    clusType=2;  

  if ( cluster.layer()==PFLayer::HF_EM) clusType=3;  

  if ( cluster.layer()==PFLayer::HCAL_BARREL2) clusType=4;  
  


  const math::XYZPoint& xyzPos = cluster.position();

  GPFCluster *gc;
        
  for (int viewType=0;viewType<NViews;viewType++){
    if (!drawHO_ && viewType==EHO) continue;
    switch(viewType) {
    case XY:
      {
        if (clusType==0) {
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.X(), xyzPos.Y(), clusPattern_);
	} else if (clusType==1) {
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.X(), xyzPos.Y(), clusPSPattern_);
             
	} else if (clusType==2) {
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.X(), xyzPos.Y(), clusPatternecal_);
	} else if (clusType==3) {
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.X(), xyzPos.Y(), clusPatternhfem_);
        } else {
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.X(), xyzPos.Y(), clusPatternho_);
        }                                     
        graphicMap_.insert(pair<int,GPFBase *> (ident, gc));
      }
      break;
    case RZ:
      {
        double sign = 1.;
        if (cos(phi0 - phi) < 0.)
          sign = -1.;
        if ( clusType==0) { 
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.z(),sign*xyzPos.Rho(),
                               clusPattern_);
        } else if (clusType==1) { 
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.z(),sign*xyzPos.Rho(),
                               clusPSPattern_);
        } else if (clusType==2) { 
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.z(),sign*xyzPos.Rho(),
                               clusPatternecal_);
        } else if (clusType==3) { 
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.z(),sign*xyzPos.Rho(),
                               clusPatternhfem_);

	} else {
          gc = new  GPFCluster(this,
                               viewType,ident,
                               &cluster,
                               xyzPos.z(),sign*xyzPos.Rho(),
                               clusPatternho_);
	}          
        graphicMap_.insert(pair<int,GPFBase *>  (ident, gc));	
      } 
      break;
    case EPE:
      {
        if( cluster.layer()<0 || cluster.layer()==PFLayer::HF_EM) {
          if (clusType==2) {
            gc = new  GPFCluster(this,
                                 viewType,ident,
                                 &cluster,
                                 eta,phi,
                                 clusPatternecal_);
          }
	  else if (clusType==3) {
            gc = new  GPFCluster(this,
                                 viewType,ident,
                                 &cluster,
                                 eta,phi,
                                 clusPatternhfem_);
          }	  
          else {
            gc = new  GPFCluster(this,
                                 viewType,ident,
                                 &cluster,
                                 eta,phi,
                                 clusPSPattern_);
          }                              
                                         
          graphicMap_.insert(pair<int,GPFBase *>        (ident, gc));
        }

      } 
      break;
    case EPH:
      {
        if( cluster.layer()>0 && cluster.layer()!=PFLayer::HF_EM  && cluster.layer()!=PFLayer::HCAL_BARREL2) {
          if (clusType==0) {
            gc = new  GPFCluster(this,
                                 viewType,ident,
                                 &cluster,
                                 eta,phi,clusPattern_);
          } 
          else {
            gc = new  GPFCluster(this,
                                 viewType,ident,
                                 &cluster,
                                 eta,phi,clusPSPattern_);
          }
            
                                         
          graphicMap_.insert(pair<int,GPFBase *>        (ident, gc));          
        }
      } 
      break;
      
    case EHO:
      {
	if( cluster.layer()>0 && cluster.layer()==PFLayer::HCAL_BARREL2) {
          if (clusType==4) {
            gc = new  GPFCluster(this,
                                 viewType,ident,
                                 &cluster,
                                 eta,phi,clusPatternho_);
          }
          else {
            gc = new  GPFCluster(this,
                                 viewType,ident,
                                 &cluster,
                                 eta,phi,clusPSPattern_);
          }
            
                                         
          graphicMap_.insert(pair<int,GPFBase *>        (ident, gc));          
        }
      } 
      break;

    default :break; 
    } 
  }      
}
//________________________________________________________________________________________
void DisplayManager::createGPart( const reco::PFSimParticle &ptc,
                                  const std::vector<reco::PFTrajectoryPoint>& points, 
                                  int ident,double pt,double phi0, double sign, bool displayInitial,
                                  int markerIndex)
{
  //bool inside = false; 
  //TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  bool debug_createGpart = false;
  //    bool debug_createGpart = true;
  for (int viewType=0;viewType<NViews;++viewType) {
    if (!drawHO_ && viewType==EHO) continue;
    //  for (int viewType=0;viewType<4;++viewType) {
    // reserving space. nb not all trajectory points are valid
    vector<double> xPos;
    xPos.reserve( points.size() );
    vector<double> yPos;
    yPos.reserve( points.size() );
                
    for(unsigned i=0; i<points.size(); i++) {

      if( !points[i].isValid() ) continue;
        
      const math::XYZPoint& xyzPos = points[i].position();      

      double eta = xyzPos.Eta();
      double phi = xyzPos.Phi();
                        
      // always take momentum for the first point 
      // otherwise decay products from resonances will have wrong eta, phi
      //calculated from position (0,0,0) (MDN 11 June 2008)
      if( points[i].layer() == reco::PFTrajectoryPoint::ClosestApproach ) {
                                
        //      if( !displayInitial && 
        //              points[i].layer() == reco::PFTrajectoryPoint::ClosestApproach ) {
        const math::XYZTLorentzVector& mom = points[i].momentum();
        eta = mom.Eta();
        phi = mom.Phi();
      }
                        
      //if( !cutg || cutg->IsInside( eta, phi ) ) 
      //  inside = true;
                        
      switch(viewType) {
      case XY:
        xPos.push_back(xyzPos.X());
        yPos.push_back(xyzPos.Y());
        break;
      case RZ:
        xPos.push_back(xyzPos.Z());
        yPos.push_back(sign*xyzPos.Rho());
        break;
      case EPE:
      case EPH:
      case EHO:
        xPos.push_back( eta );
        yPos.push_back( phi );
        break;
      default:break;
      }
    }  
    if (viewType == EPE && debug_createGpart) {
      cout << " display PFsim eta/phi view ECAL" << endl;
      cout << " nb of points for display " << xPos.size() << endl;
      for(unsigned i=0; i<xPos.size(); i++) {
        cout << " point " << i << " x/y " << xPos[i] <<"/" << yPos[i]<< endl;
      }
    }
                
    /// no point inside graphical cut.
    //if( !inside ) return;
    //int color = 4;
    GPFSimParticle *gc   = new GPFSimParticle(this,
                                              viewType, ident,
                                              &ptc,
                                              xPos.size(),&xPos[0],&yPos[0],
                                              pt,
                                              simPartPatternM_[markerIndex],
                                              simPartPatternL_,
                                              "pl");
                                              
    graphicMap_.insert(pair<int,GPFBase *>      (ident, gc));                                 
    //graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFSimParticle(this,viewType,ident,&ptc,xPos.size(),&xPos[0],&yPos[0],pt,simPartPattern_[indexMarker], "pl")));
  }
}
//____________________________________________________________________
void DisplayManager::createGRecHit(reco::PFRecHit& rh,int ident, double maxe, double phi0, int color)
{
        
  double me = maxe;
  double thresh = 0;
  int layer = rh.layer();
        
        
  switch(layer) {
  case PFLayer::ECAL_BARREL:
    thresh = em_->clusterAlgoECAL_.threshBarrel();
    break;     
  case PFLayer::ECAL_ENDCAP:
    thresh = em_->clusterAlgoECAL_.threshEndcap();
    break;     
  case PFLayer::HCAL_BARREL1:
  case PFLayer::HCAL_BARREL2:
    thresh = em_->clusterAlgoHCAL_.threshBarrel();
    break;           
  case PFLayer::HCAL_ENDCAP:
    thresh = em_->clusterAlgoHCAL_.threshEndcap();
    break;                     
  case PFLayer::HF_HAD: 
    // how to handle a different threshold for HF?
    thresh = em_->clusterAlgoHFHAD_.threshEndcap();
    break;           
  case PFLayer::HF_EM: 
    // how to handle a different threshold for HF?
    thresh = em_->clusterAlgoHFEM_.threshEndcap();
    break;           
  case PFLayer::PS1:
  case PFLayer::PS2:
    me = -1;
    thresh = em_->clusterAlgoPS_.threshBarrel();
    break;
  default:
    {
      cerr<<"DisplayManager::createGRecHit : manage other layers."
          <<" GRechit notcreated."<<endl;
      return;
    }
  }
  if( rh.energy() < thresh ) return;
        
  //loop on all views
  for(int viewType=0;viewType<NViews;++viewType) {
    if (!drawHO_ && viewType==EHO) continue;

    bool isHCAL = (layer == PFLayer::HCAL_BARREL1 || 
		   //                   layer == PFLayer::HCAL_BARREL2 || 
                   layer == PFLayer::HCAL_ENDCAP || 
                   layer == PFLayer::HF_HAD);
    
    if(  viewType == EPH && 
         ! isHCAL) {
      continue;
    }
    // on EPE view, draw only HCAL and preshower
    if(  viewType == EPE && (isHCAL || layer ==PFLayer::HCAL_BARREL2)) {
      continue;
    }

    if (viewType == EHO && layer !=PFLayer::HCAL_BARREL2) continue;


    double rheta = rh.position().Eta();
    double rhphi = rh.position().Phi();

    double sign = 1.;
    if (cos(phi0 - rhphi) < 0.) sign = -1.;
                
                
    double etaSize[4];
    double phiSize[4];
    double x[5];
    double y[5];
    double z[5];
    double r[5];
    double eta[5];
    double phi[5];
    double xprop[5];
    double yprop[5];
    double etaprop[5];
    double phiprop[5];
                
                
    const std::vector< math::XYZPoint >& corners = rh.getCornersXYZ();
    assert(corners.size() == 4);
    double propfact = 0.95; // so that the cells don't overlap ? 
    double ampl=0;
    if(me>0) ampl = (log(rh.energy() + 1.)/log(me + 1.));
    for ( unsigned jc=0; jc<4; ++jc ) { 

      phiSize[jc] = rhphi-corners[jc].Phi();
      etaSize[jc] = rheta-corners[jc].Eta();
      if ( phiSize[jc] > 1. ) phiSize[jc] -= 2.*TMath::Pi();  
      if ( phiSize[jc] < -1. ) phiSize[jc]+= 2.*TMath::Pi();

      phiSize[jc] *= propfact;
      etaSize[jc] *= propfact;
                        
      math::XYZPoint cornerposxyz = corners[jc];
                        
      x[jc] = cornerposxyz.X();
      y[jc] = cornerposxyz.Y();
      z[jc] = cornerposxyz.Z();
      r[jc] = sign*cornerposxyz.Rho();
      eta[jc] = rheta - etaSize[jc];
      phi[jc] = rhphi - phiSize[jc];
                        
                        
      // cell area is prop to log(E)
      // not drawn for preshower. 
      // otherwise, drawn for eta/phi view, and for endcaps in xy view
      if( layer != PFLayer::PS1 && 
          layer != PFLayer::PS2 && 
          ( viewType == EPE || 
            viewType == EPH || 
	    viewType == EHO || 
            ( viewType == XY &&  
              ( layer == PFLayer::ECAL_ENDCAP || 
                layer == PFLayer::HCAL_ENDCAP || 
                layer == PFLayer::HF_HAD
                ) ) ) ) {
      
	math::XYZPoint centreXYZrot = rh.position();
	
        math::XYZPoint centertocorner(x[jc] - centreXYZrot.X(), 
                                      y[jc] - centreXYZrot.Y(),
                                      0 );
	
        math::XYZPoint centertocornerep(eta[jc] - centreXYZrot.Eta(), 
                                        phi[jc] - centreXYZrot.Phi(),
                                        0 );
	
        
        // centertocorner -= centreXYZrot;
        xprop[jc] = centreXYZrot.X() + centertocorner.X()*ampl;
        yprop[jc] = centreXYZrot.Y() + centertocorner.Y()*ampl;
                                
        etaprop[jc] = centreXYZrot.Eta() + centertocornerep.X()*ampl;
        phiprop[jc] = centreXYZrot.Phi() + centertocornerep.Y()*ampl;
      }
    }//loop on jc 
                
    if(layer == PFLayer::ECAL_BARREL  || 
       layer == PFLayer::HCAL_BARREL1 || 
       layer == PFLayer::HCAL_BARREL2 || viewType == RZ) {
                        
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
      
      double signx=1.;
      if(layer == PFLayer::HCAL_BARREL2) { signx=-1.;}
                        
      x[i1] *= 1+signx*ampl/2.;
      x[i2] *= 1+signx*ampl/2.;
      y[i1] *= 1+signx*ampl/2.;
      y[i2] *= 1+signx*ampl/2.;
      z[i1] *= 1+signx*ampl/2.;
      z[i2] *= 1+signx*ampl/2.;
      r[i1] *= 1+signx*ampl/2.;
      r[i2] *= 1+signx*ampl/2.;

    }

    x[4]=x[0];
    y[4]=y[0]; // closing the polycell
    z[4]=z[0];
    r[4]=r[0]; // closing the polycell
    eta[4]=eta[0];
    phi[4]=phi[0]; // closing the polycell
                
    int npoints=5;
                
    switch( viewType ) {
    case  XY:
      {
        if(layer == PFLayer::ECAL_BARREL || 
           layer == PFLayer::HCAL_BARREL1 || 
           layer == PFLayer::HCAL_BARREL2) {
          graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,x,y,color,"f")));
                                        
        } else {
          graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,x,y,color,"l")));
          if( ampl>0 ) { // not for preshower
            xprop[4]=xprop[0];
            yprop[4]=yprop[0]; // closing the polycell    
            graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,xprop,yprop,color,"f")));
          }
        }
      } 
      break;
                                
    case RZ:
      graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,z,r,color,"f")));
      break;
                                
    case EPE:
      {
        graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,eta,phi,color,"l")));
                                
        if( ampl>0 ) { // not for preshower
          etaprop[4]=etaprop[0];
          phiprop[4]=phiprop[0]; // closing the polycell    
          graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,etaprop,phiprop,color,"f")));
        }
      }  
      break;
    case EPH:
      {      
	if(layer == PFLayer::ECAL_BARREL || 
           layer == PFLayer::HCAL_BARREL1) {
	  graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,eta,phi,color,"l")));
	  
	  if( ampl>0 ) { // not for preshower
	    etaprop[4]=etaprop[0];
	    phiprop[4]=phiprop[0]; // closing the polycell    
	    graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,etaprop,phiprop,color,"f")));
	  }
	} 
      }
      break;

    case EHO:
      {
	
        graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,eta,phi,color,"l")));
        if(layer == PFLayer::HCAL_BARREL2) {                        
	  if( ampl>0 ) { // not for preshower
	    etaprop[4]=etaprop[0];
	    phiprop[4]=phiprop[0]; // closing the polycell    
	    graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,etaprop,phiprop,color,"f")));
	  }
	} 
      }
      break;


                                
    default: break;
    }//switch end
                
  } //loop on views
}

//_________________________________________________________________________________________
void DisplayManager::createGTrack( reco::PFRecTrack &tr,
                                   const std::vector<reco::PFTrajectoryPoint>& points, 
                                   int ident,double pt,double phi0, double sign, bool displayInitial,
                                   int linestyle,int kfgsfbrem) 
{
        
  //   bool inside = false; 
  //TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
        
  for (int viewType=0;viewType<NViews;++viewType) {
    if (!drawHO_ && viewType==EHO) continue;
    // reserving space. nb not all trajectory points are valid
    vector<double> xPos;
    xPos.reserve( points.size() );
    vector<double> yPos;
    yPos.reserve( points.size() );
                
    for(unsigned i=0; i<points.size(); i++) {
      if( !points[i].isValid() ) continue;
      
      const math::XYZPoint& xyzPos = points[i].position();
      //muriel      
      //if(kfgsfbrem) 
      //   std::cout << xyzPos << std::endl;
      double eta = xyzPos.Eta();
      double phi = xyzPos.Phi();
                        
      if( !displayInitial && 
          points[i].layer() == reco::PFTrajectoryPoint::ClosestApproach ) {
        const math::XYZTLorentzVector& mom = points[i].momentum();
        eta = mom.Eta();
        phi = mom.Phi();
      }
                        
      //if( !cutg || cutg->IsInside( eta, phi ) ) 
      //  inside = true;
                        
                        
      switch(viewType) {
      case XY:
        xPos.push_back(xyzPos.X());
        yPos.push_back(xyzPos.Y());
        break;
      case RZ:
        xPos.push_back(xyzPos.Z());
        yPos.push_back(sign*xyzPos.Rho());
        break;
      case EPE:
      case EPH:
      case EHO:	
        xPos.push_back( eta );
        yPos.push_back( phi );
        break;
      }
    }  
    /// no point inside graphical cut.
    //if( !inside ) return;
                
    //fill map with graphic objects
      GPFTrack *gt=0;
      if(kfgsfbrem==0) {
	gt = new  GPFTrack(this,
			   viewType,ident,
			   &tr,
			   xPos.size(),&xPos[0],&yPos[0],pt,
			   trackPatternM_,trackPatternL_,"pl");
      }
      else if (kfgsfbrem==1) {
	//	std::cout << " Creating the GSF track " << std::endl;
	gt = new  GPFTrack(this,
			   viewType,ident,
			   &tr,
			   xPos.size(),&xPos[0],&yPos[0],pt,
			   gsfPatternM_,gsfPatternL_,"pl");
      }
      else if (kfgsfbrem==2) {
	//	std::cout << " Creating the Brem " << std::endl;
	//	std::cout<<"adr brem dans create:"<<&tr<<std::flush<<std::endl;
	gt = new  GPFTrack(this,
			   viewType,ident,
			   &tr,
			   xPos.size(),&xPos[0],&yPos[0],pt,
			   bremPatternM_,bremPatternL_,"pl");
      }
      graphicMap_.insert(pair<int,GPFBase *> (ident, gt));
  }   
}


void DisplayManager::displayEvent(int run, int lumi, int event) {
  reset();
  em_->processEvent(run, lumi, event);  
  eventNumber_= em_->eventNumber();
  loadGraphicObjects();
  isGraphicLoaded_= true;
  displayAll();
}


//________________________________________________________
void DisplayManager::display(int ientry)
{
  if (ientry<0 || ientry>maxEvents_) {
    std::cerr<<"DisplayManager::no event matching criteria"<<std::endl;
    return;
  }  
  reset();
  em_->processEntry(ientry);  
  eventNumber_= em_->eventNumber();
  loadGraphicObjects();
  isGraphicLoaded_= true;

  displayAll();

}
//________________________________________________________________________________
void DisplayManager::displayAll(bool noRedraw)
{

  if (!isGraphicLoaded_) {
    std::cout<<" no Graphic Objects to draw"<<std::endl;
    return;
  }
  if (noRedraw) { 
    for (int viewType=0;viewType<NViews;++viewType) {
      if (!drawHO_ && viewType==EHO) continue;
      displayView_[viewType]->cd();
      gPad->Clear();
    } 
    //TODOLIST: add test on view to draw 
    displayCanvas();
  }  
  
  std::multimap<int,GPFBase *>::iterator p;

  for (p=graphicMap_.begin();p!=graphicMap_.end();p++) {
    int ident=p->first;
    int type=ident >> shiftId_;
    int view = p->second->getView();
    
    if (!drawHO_ && (view ==EHO || type==CLUSTERHOID || type==RECHITHOID)) continue;
    
    switch (type) {
    case CLUSTERECALID: 
    case CLUSTERHCALID: 
    case CLUSTERHOID: 
    case CLUSTERHFEMID: 
    case CLUSTERHFHADID: 
    case CLUSTERPSID: 
    case CLUSTERIBID:
      {
	//	cout<<"displaying "<<type<<" "<<p->second->getEnergy()<<" "<<clusEnMin_<<" on view "<<view<<endl;
        if (drawClus_)
          if (p->second->getEnergy() > clusEnMin_) {
            displayView_[view]->cd();
            p->second->draw();
          }        
      }
      break;
    case RECHITECALID: 
    case RECHITHCALID: 
    case RECHITHOID: 
    case RECHITHFEMID: 
    case RECHITHFHADID: 
    case RECHITPSID:
      {
        if (!noRedraw) break; 
        if (drawHits_) 
          if(p->second->getEnergy() > hitEnMin_)  {
            displayView_[view]->cd();
            p->second->draw();
          }
        break;  
      }  
    case RECTRACKID:
      {
        if (drawTracks_) 
          if (p->second->getPt() > trackPtMin_) {
            displayView_[view]->cd();
            p->second->draw();
          }
      }
      break;
    case GSFRECTRACKID:
      {
        if (drawGsfTracks_) 
          if (p->second->getPt() > gsfPtMin_) {
            displayView_[view]->cd();
            p->second->draw();
          }
      }
      break;
    case BREMID:
      {
	if (drawBrems_)
	  {
	    displayView_[view]->cd();
	    p->second->draw();
	  }
      }
      break;
    case SIMPARTICLEID:
      {
        if (drawParticles_)
          if (p->second->getPt() > particlePtMin_) {
            displayView_[view]->cd();
            p->second->draw();
          }
      }
      break;
    case GENPARTICLEID:
      {  
        if (drawGenParticles_) 
          if (p->second->getPt() > genParticlePtMin_) 
            if (view == EPH || view ==EPE || view ==EHO) {
              displayView_[view]->cd();
              p->second->draw();
            }  
      } 
      break;           
    default : std::cout<<"DisplayManager::displayAll()-- unknown object "<<std::endl;               
    }  //switch end
  }   //for end

  for (int i=0;i<NViews;i++) {
    if (!drawHO_ && i==EHO) continue;
    displayView_[i]->cd();
    gPad->Modified();
    displayView_[i]->Update();
  }  
}
//___________________________________________________________________________________
void DisplayManager::drawWithNewGraphicAttributes()
{
  std::multimap<int,GPFBase *>::iterator p;
        
  for (p=graphicMap_.begin();p!=graphicMap_.end();p++) {
    int ident=p->first;
    int type=ident >> shiftId_;
    if (!drawHO_ && (type==CLUSTERHOID)) continue;
    switch (type) {
    case CLUSTERECALID: case CLUSTERHCALID:  case CLUSTERHOID: case  CLUSTERPSID: case CLUSTERIBID:
      {
        p->second->setNewStyle();
        p->second->setNewSize();
        p->second->setColor();  
      }
      break;
    case RECTRACKID:
      {
        p->second->setColor();
        p->second->setNewStyle();
        p->second->setNewSize();
      }
      break;
    case SIMPARTICLEID:
      {
      }
      break;
    default : break;            
    }  //switch end
  }   //for end
  displayAll(false);
}
//___________________________________________________________________________________
void DisplayManager::displayCanvas()
{
  double zLow = -500.;
  double zUp  = +500.;
  double rUp  = +400.;
  if (!drawHO_) rUp=300.0;
  double etarng=(drawHO_) ? 1.19 : 1.39;
  double ratio =(drawHO_) ? 1.15 : 0.90;
  double scal = (drawHO_) ? 1.00 : 0.85;
        
  //TODOLIST : test wether view on/off
  //if (!displayView_[viewType] || !gROOT->GetListOfCanvases()->FindObject(displayView_[viewType]) ) {
  //   assert(viewSize_.size() == 2);
        


  for (int viewType=0;viewType<NViews;++viewType) {
    if (!drawHO_ && viewType==EHO) continue;
    displayView_[viewType]->cd();
    displayHist_[viewType]->Draw();
    switch(viewType) {
    case XY: 
      frontFaceECALXY_.Draw();
      frontFaceHCALXY_.Draw();
      frontFaceHOXY_.Draw();
      break;
    case RZ:    
      {// Draw lines at different etas
        TLine l;
        l.SetLineColor(1);
        l.SetLineStyle(3);
        TLatex etaLeg;
        etaLeg.SetTextSize( (drawHO_) ? 0.03 : 0.02);
        float etaMin = -3.;
        float etaMax = +3.;
        float etaBin = 0.2;
        int nEtas = int((etaMax - etaMin)/0.2) + 1;
        for (int iEta = 0; iEta <= nEtas; iEta++) {
          float eta = etaMin + iEta*etaBin;
          float r = ratio*rUp;
          TVector3 etaImpact;
          etaImpact.SetPtEtaPhi(r, eta, 0.);
          etaLeg.SetTextAlign(21);
          if (eta <= -etarng) {
            etaImpact.SetXYZ(0.,scal*zLow*tan(etaImpact.Theta()),scal*zLow);
            etaLeg.SetTextAlign(31);
          } else if (eta >= etarng) {
            etaImpact.SetXYZ(0.,scal*zUp*tan(etaImpact.Theta()),scal*zUp);
            etaLeg.SetTextAlign(11);
          }
	  if (drawHO_) {
	    if (fabs(eta)<1.29) {
	      etaLeg.SetTextSize(0.030);
	    } else {
	      etaLeg.SetTextSize(0.022);
	    }
	  }
          l.DrawLine(0., 0., etaImpact.Z(), etaImpact.Perp());
          etaLeg.DrawLatex(1.0*etaImpact.Z(), etaImpact.Perp(), Form("%2.1f", eta));
        }
        frontFaceECALRZ_.Draw();
      } 
      break;
    default: break;
    } //end switch
  }
}
//________________________________________________________________________________
void DisplayManager::displayNext()
{
  int eventNumber_=em_->eventNumber();
  display(++eventNumber_);
}
//_________________________________________________________________________________
void DisplayManager::displayNextInteresting(int ientry)
{
  bool ok=false;
  while (!ok && ientry<em_->ev_->size() ) {
    ok = em_->processEntry(ientry);
    ientry++;
  }
  eventNumber_ = em_->eventNumber();
  if (ok) {
    reset();
    loadGraphicObjects();
    isGraphicLoaded_= true;
    displayAll();
  }
  else 
    std::cerr<<"DisplayManager::dislayNextInteresting : no event matching criteria"<<std::endl;
}
//_________________________________________________________________________________
void DisplayManager::displayPrevious()
{
  int eventNumber_=em_->eventNumber();
  display(--eventNumber_);
}
//______________________________________________________________________________
void DisplayManager::rubOutGPFBlock()
{
  int size = selectedGObj_.size();
  bool toInitial=true;
  int color=0;
  for (int i=0;i<size;i++) 
    drawGObject(selectedGObj_[i],color,toInitial);
}
//_______________________________________________________________________________
void DisplayManager::displayPFBlock(int blockNb) 
{
  rubOutGPFBlock();
  selectedGObj_.clear();
  if (!drawPFBlocks_) return;
  int color=1;
  multimap<int,pair <int,int > >::const_iterator p;
  p= blockIdentsMap_.find(blockNb);
  if (p !=blockIdentsMap_.end()) {
    do {
      int ident=(p->second).first;
      drawGObject(ident,color,false);
      p++;
    } while (p!=blockIdentsMap_.upper_bound(blockNb));
  }
  else 
    cout<<"DisplayManager::displayPFBlock :not found"<<endl;    
}  
//_______________________________________________________________________________
void DisplayManager::drawGObject(int ident,int color,bool toInitial) 
{
  typedef std::multimap<int,GPFBase *>::const_iterator iter;
  iter p;
  std::pair<iter, iter > result = graphicMap_.equal_range(ident);
  if(result.first == graphicMap_.end()) {
    std::cout<<"pas d'objet avec cet ident: "<<ident<<std::flush<<std::endl; 
    return;
  }
  p=result.first;
  while (p != result.second) {
    int view=p->second->getView();
    if (!drawHO_ && view ==EHO) continue; 
    displayView_[view]->cd();
    if (toInitial) p->second->setInitialColor();
    else p->second->setColor(color);
    p->second->draw();
    gPad->Modified();
    //      displayView_[view]->Update();
    if (!toInitial) selectedGObj_.push_back(ident);
    p++; 
  }
}
//______________________________________________________________________________
void DisplayManager::enableDrawPFBlock(bool state)
{
  drawPFBlocks_=state;
}
//______________________________________________________________________________
void DisplayManager::enableDrawBrem(bool state)
{
  drawBrems_=state;
}  
//_______________________________________________________________________________
void DisplayManager::findAndDraw(int ident) 
{       
  int type=ident >> shiftId_;            // defined in DisplayCommon.h
  if (drawHO_ || type!=RECHITHOID) {
    int color=1;
    if (type>15) {
      std ::cout<<"DisplayManager::findAndDraw :object Type unknown"<<std::endl;
      return;
    }  
    if (drawPFBlocks_==0  ||
	type==RECHITECALID || type==RECHITHCALID  || type==RECHITHOID  || 
	type==RECHITHFEMID || type==RECHITHFHADID ||
	type==RECHITPSID   || type==SIMPARTICLEID)   {
      rubOutGPFBlock();
      selectedGObj_.clear();
      bool toInitial=false;
      drawGObject(ident,color,toInitial);
      if (type<HITTYPES) {
	//redrawWithoutHits_=true;
	displayAll(false);
	//redrawWithoutHits_=false;
      }
    }     
    updateDisplay();
  }
}
//___________________________________________________________________________________
void DisplayManager::findBlock(int ident) 
{
  int blockNb=-1;
  int elemNb=-1;
  multimap<int, pair <int,int > >::const_iterator p;
  for (p=blockIdentsMap_.begin();p!=blockIdentsMap_.end();p++) {
    int id=(p->second).first;
    if (id == ident) {
      blockNb=p->first;
      elemNb=(p->second).second;
      break;
    }   
  }
  if (blockNb > -1) {
    std::cout<<"this object is element "<<elemNb<<" of PFblock nb "<<blockNb<<std::endl;
    assert( blockNb < static_cast<int>(em_->blocks().size()) );
    const reco::PFBlock& block = em_->blocks()[blockNb];
    std::cout<<block<<std::endl;
    displayPFBlock(blockNb);
  }   
  updateDisplay();
}
//_______________________________________________________________________ 
bool DisplayManager::findBadBremsId(int ident)
{
  for (unsigned i=0;i<badBremsId_.size();i++)
    if (badBremsId_[i]==ident) return true;
  return false;  
}
//_________________________________________________________________________

void DisplayManager::updateDisplay() {
  for(unsigned i=0; i<displayView_.size(); i++) {
    if (!drawHO_ && i==EHO) continue;
    TPad* p =  displayView_[i];
    assert( p );
    p->Modified();
    p->Update();
  }
}


//_________________________________________________________________________
double DisplayManager::getMaxE(int layer) const
{
        
  double maxe = -9999;
        
  // in which vector should we look for these rechits ?
        
  const reco::PFRecHitCollection* vec = 0;
  switch(layer) {
  case PFLayer::ECAL_ENDCAP:
  case PFLayer::ECAL_BARREL:
    vec = &(em_->rechitsECAL_);
    break;
  case PFLayer::HCAL_ENDCAP:
  case PFLayer::HCAL_BARREL1:
    //  case PFLayer::HCAL_BARREL2:
    vec = &(em_->rechitsHCAL_);
    break;
  case PFLayer::HCAL_BARREL2:
    vec = &(em_->rechitsHO_);
    break;

  case PFLayer::HF_EM: 
    vec = &(em_->rechitsHFEM_);
    break;
  case PFLayer::HF_HAD: 
    vec = &(em_->rechitsHFHAD_);
    break;
  case PFLayer::PS1:
  case PFLayer::PS2:
    vec = &(em_->rechitsPS_);
    break;
  default:
    cerr<<"DisplayManager::getMaxE : manage other layers"<<endl;
    return maxe;
  }
        
  for( unsigned i=0; i<vec->size(); i++) {
    if( (*vec)[i].layer() != layer ) continue;
    if( (*vec)[i].energy() > maxe)
      maxe = (*vec)[i].energy();
  }
        
  return maxe;
}
//____________________________________________________________________________
double DisplayManager::getMaxEEcal() {
        
  if( maxERecHitEcal_<0 ) {
    double maxeec = getMaxE( PFLayer::ECAL_ENDCAP );
    double maxeb =  getMaxE( PFLayer::ECAL_BARREL );
    double maxehf =  getMaxE( PFLayer::HF_EM );
    maxERecHitEcal_ =  maxeec>maxeb  ?  maxeec:maxeb;
    maxERecHitEcal_ = maxERecHitEcal_>maxehf ? maxERecHitEcal_:maxehf;
    // max of both barrel and endcap
  }
  return  maxERecHitEcal_;
}
//_______________________________________________________________________________
double DisplayManager::getMaxEHcal() {
        
  if(maxERecHitHcal_ < 0) {
    double maxehf = getMaxE( PFLayer::HF_HAD );
    double maxeec = getMaxE( PFLayer::HCAL_ENDCAP );
    double maxeb =  getMaxE( PFLayer::HCAL_BARREL1 );
    maxERecHitHcal_ =  maxeec>maxeb  ?  maxeec:maxeb;
    maxERecHitHcal_ = maxERecHitHcal_>maxehf ? maxERecHitHcal_:maxehf;
  }
  return maxERecHitHcal_;
} 

//_______________________________________________________________________________
double DisplayManager::getMaxEHo() {
        
  if(maxERecHitHo_ < 0) {
    maxERecHitHo_ =  getMaxE( PFLayer::HCAL_BARREL2 );
  }
  return maxERecHitHo_;
} 


//________________________________________________________________________________________________
void DisplayManager::loadGGenParticles()
{
  
  const HepMC::GenEvent* myGenEvent = em_->MCTruth_.GetEvent();
  if(!myGenEvent) return;
  for ( HepMC::GenEvent::particle_const_iterator piter  = myGenEvent->particles_begin();
        piter != myGenEvent->particles_end(); 
        ++piter ) {
    HepMC::GenParticle* p = *piter;

    createGGenParticle(p);
  } 
}
//____________________________________________________________________________
void DisplayManager::createGGenParticle(HepMC::GenParticle* p)
{
  // these are the beam protons
  if ( !p->production_vertex() && p->pdg_id() == 2212 ) return;
    
  int partId = p->pdg_id();

  std::string name;
  std::string latexStringName;

  name = em_->getGenParticleName(partId,latexStringName);
  int barcode = p->barcode();
  int genPartId=(GENPARTICLEID<<shiftId_) | barcode;
    
    
//   int vertexId1 = 0;
//   vertexId1 = p->production_vertex()->barcode();
    
//   math::XYZVector vertex1 (p->production_vertex()->position().x(),
//                            p->production_vertex()->position().y(),
//                            p->production_vertex()->position().z());
                             
  math::XYZTLorentzVector momentum1(p->momentum().px(),
                                    p->momentum().py(),
                                    p->momentum().pz(),
                                    p->momentum().e());
                                      
  double eta = momentum1.eta();
  if ( eta > +10. ) eta = +10.;
  if ( eta < -10. ) eta = -10.;
    
  double phi = momentum1.phi();
    
  double pt = momentum1.pt();
  double e = momentum1.e();
    
  //mother ?    

  // Colin: the following line gives a segmentation fault when there is 
  // no particles entering the production vertex.
    
  // const HepMC::GenParticle* mother = 
  //  *(p->production_vertex()->particles_in_const_begin());
    
  // protecting against this in the following way:
  const HepMC::GenParticle* mother = 0;
  if(p->production_vertex() && 
     p->production_vertex()->particles_in_size() ) {
    mother = 
      *(p->production_vertex()->particles_in_const_begin()); 
  }

  // Colin: no need to declare this pointer in this context.
  // the declaration can be more local.
  // GPFGenParticle *gp; 

  int modnViews = (drawHO_) ? NViews : NViews-1;
  if ( mother ) {
    int barcodeMother = mother->barcode();
    math::XYZTLorentzVector momentumMother(mother->momentum().px(),
                                           mother->momentum().py(),
                                           mother->momentum().pz(),
                                           mother->momentum().e());
    double etaMother = momentumMother.eta();                                  
    if ( etaMother > +10. ) etaMother = +10.;
    if ( etaMother < -10. ) etaMother = -10.;
    double phiMother = momentumMother.phi();
    
       
    double x[2],y[2];
    

    x[0]=etaMother;x[1]=eta;
    y[0]=phiMother;y[1]=phi;
       
    for (int view = 2; view< modnViews; view++) {
      GPFGenParticle* gp   = new GPFGenParticle(this,             
                                                view, genPartId,
                                                x, y,              //double *, double *
                                                e,pt,barcode,barcodeMother,
                                                genPartPattern_, 
                                                name,latexStringName);
      graphicMap_.insert(pair<int,GPFBase *>    (genPartId, gp));
    }
  }
  else {     //no Mother    
    for (int view = 2; view< modnViews; view++) {
      GPFGenParticle* gp   = new GPFGenParticle(this,
                                                view, genPartId,
                                                eta, phi,                  //double double
                                                e,pt,barcode,
                                                genPartPattern_,
                                                name, latexStringName);
      graphicMap_.insert(pair<int,GPFBase *>    (genPartId, gp));
    }                                         
  }
}
//____________________________________________________________________________  
void DisplayManager::loadGClusters()
{
  double phi0=0;
        
  for(unsigned i=0; i<em_->clustersECAL_->size(); i++){
    //int clusId=(i<<shiftId_) | CLUSTERECALID;
    int clusId=(CLUSTERECALID<<shiftId_) | i;
    createGCluster( (*(em_->clustersECAL_))[i],clusId, phi0);
  }    
  for(unsigned i=0; i<em_->clustersHCAL_->size(); i++) {
    //int clusId=(i<<shiftId_) | CLUSTERHCALID;
    int clusId=(CLUSTERHCALID<<shiftId_) | i;
    createGCluster( (*(em_->clustersHCAL_))[i],clusId, phi0);
  }    

  if (drawHO_) {
    for(unsigned i=0; i<em_->clustersHO_->size(); i++) {
      //int clusId=(i<<shiftId_) | CLUSTERHOID;
      int clusId=(CLUSTERHOID<<shiftId_) | i;
      createGCluster( (*(em_->clustersHO_))[i],clusId, phi0);
    }
  }

  for(unsigned i=0; i<em_->clustersHFEM_->size(); i++) {
    //int clusId=(i<<shiftId_) | CLUSTERHFEMID;
    int clusId=(CLUSTERHFEMID<<shiftId_) | i;
    createGCluster( (*(em_->clustersHFEM_))[i],clusId, phi0);
  }    
  for(unsigned i=0; i<em_->clustersHFHAD_->size(); i++) {
    //int clusId=(i<<shiftId_) | CLUSTERHFHADID;
    int clusId=(CLUSTERHFHADID<<shiftId_) | i;
    createGCluster( (*(em_->clustersHFHAD_))[i],clusId, phi0);
  }    
  for(unsigned i=0; i<em_->clustersPS_->size(); i++){ 
    //int clusId=(i<<shiftId_) | CLUSTERPSID;
    int clusId=(CLUSTERPSID<<shiftId_) | i;
    createGCluster( (*(em_->clustersPS_))[i],clusId,phi0);
  }
//   for(unsigned i=0; i<em_->clustersIslandBarrel_.size(); i++) {
//     PFLayer::Layer layer = PFLayer::ECAL_BARREL;
//     //int clusId=(i<<shiftId_) | CLUSTERIBID;
//     int clusId=(CLUSTERIBID<<shiftId_) | i;
                
//     reco::PFCluster cluster( layer, 
//                              em_->clustersIslandBarrel_[i].energy(),
//                              em_->clustersIslandBarrel_[i].x(),
//                              em_->clustersIslandBarrel_[i].y(),
//                              em_->clustersIslandBarrel_[i].z() ); 
//     createGCluster( cluster,clusId, phi0);
//   }  
}
//_____________________________________________________________________
void DisplayManager::retrieveBadBrems()
{

  //selects Brems with no information in PFBlock.Those selected Brems are not displayed
   
  int size = em_->pfBlocks_->size();
  for (int ibl=0;ibl<size;ibl++) {
     edm::OwnVector< reco::PFBlockElement >::const_iterator iter;
    for( iter =((*(em_->pfBlocks_))[ibl].elements()).begin();
         iter != ((*(em_->pfBlocks_))[ibl].elements()).end();iter++) {
       int ident=-1;  
       reco::PFBlockElement::Type type = (*iter).type();
       if (type == reco::PFBlockElement::BREM) {
          std::multimap<double, unsigned> ecalElems;
	   
	  (*(em_->pfBlocks_))[ibl].associatedElements( (*iter).index(),(*(em_->pfBlocks_))[ibl].linkData(),
				                        ecalElems ,
				                        reco::PFBlockElement::ECAL,
				                        reco::PFBlock::LINKTEST_ALL );

       
          if (ecalElems.size()==0) {
	    //             std::cout<<" PfBlock Nb "<<ibl<<" -- brem elem  "<<(*iter).index()<<"-type "<<(*iter).type()<<" not drawn"<<std::flush<<std::endl;
       	     const reco::PFBlockElementBrem * Brem =  dynamic_cast<const reco::PFBlockElementBrem*>(&(*iter)); 
	     reco::GsfPFRecTrackRef trackref = Brem->GsftrackRefPF();
	     unsigned ind=trackref.key()*40+Brem->indTrajPoint();
	     ident = (BREMID << shiftId_ ) | ind ;
	     badBremsId_.push_back(ident); 
          }
       }   
    }     
  }
}
//_____________________________________________________________________
void DisplayManager::loadGPFBlocks()
{
  int size = em_->pfBlocks_->size();
  for (int ibl=0;ibl<size;ibl++) {
    //int elemNb=((*(em_->pfBlocks_))[ibl].elements()).size();
    //std::cout<<"block "<<ibl<<":"<<elemNb<<" elements"<<std::flush<<std::endl;
    edm::OwnVector< reco::PFBlockElement >::const_iterator iter;
    for( iter =((*(em_->pfBlocks_))[ibl].elements()).begin();
         iter != ((*(em_->pfBlocks_))[ibl].elements()).end();iter++) {

      //COLIN
//       std::cout<<"elem index "<<(*iter).index()<<"-type:"
//            <<(*iter).type()<<std::flush<<std::endl;
      int ident=-1;  
       
      reco::PFBlockElement::Type type = (*iter).type();
      if (!drawHO_ && type==reco::PFBlockElement::HO) continue;
      switch (type) {
      case reco::PFBlockElement::NONE :
	assert(0);
        break;
      case reco::PFBlockElement::TRACK:
        {
          reco::PFRecTrackRef trackref =(*iter).trackRefPF();  
          assert( !trackref.isNull() );
          // std::cout<<" - key "<<trackref.key()<<std::flush<<std::endl<<std::endl;
          ident=(RECTRACKID <<shiftId_) | trackref.key();
        }
      break;
      case reco::PFBlockElement::PS1:
        {
          reco::PFClusterRef clusref=(*iter).clusterRef();
          assert( !clusref.isNull() );
          //std::cout<<"- key "<<clusref.key()<<std::flush<<std::endl<<std::endl;
          ident=(CLUSTERPSID <<shiftId_) |clusref.key();
        }
      break;
      case reco::PFBlockElement::PS2:
        {
          reco::PFClusterRef clusref=(*iter).clusterRef();
          assert( !clusref.isNull() );
          //std::cout<<"key "<<clusref.key()<<std::flush<<std::endl<<std::endl;
          ident=(CLUSTERPSID <<shiftId_) |clusref.key();
        }
      break;
      case reco::PFBlockElement::ECAL:
        {
          reco::PFClusterRef clusref=(*iter).clusterRef();
          assert( !clusref.isNull() );
          //std::cout<<"key "<<clusref.key()<<std::flush<<std::endl<<std::endl;
          ident=(CLUSTERECALID <<shiftId_) |clusref.key();
        }
      break;
      case reco::PFBlockElement::HCAL:
        {
          reco::PFClusterRef clusref=(*iter).clusterRef();
          assert( !clusref.isNull() );
          //std::cout<<"key "<<clusref.key()<<std::flush<<std::endl<<std::endl;
          ident=(CLUSTERHCALID <<shiftId_) |clusref.key();
        }
      break;

      case reco::PFBlockElement::HO:
        {
          reco::PFClusterRef clusref=(*iter).clusterRef();
          assert( !clusref.isNull() );
          //std::cout<<"key "<<clusref.key()<<std::flush<<std::endl<<std::endl;
          ident=(CLUSTERHOID <<shiftId_) |clusref.key();
        }
      break;

      case reco::PFBlockElement::HFEM:
        {
          reco::PFClusterRef clusref=(*iter).clusterRef();
          assert( !clusref.isNull() );
          //std::cout<<"key "<<clusref.key()<<std::flush<<std::endl<<std::endl;
          ident=(CLUSTERHFEMID <<shiftId_) |clusref.key();
        }
      break;
      case reco::PFBlockElement::HFHAD:
        {
          reco::PFClusterRef clusref=(*iter).clusterRef();
          assert( !clusref.isNull() );
          //std::cout<<"key "<<clusref.key()<<std::flush<<std::endl<<std::endl;
          ident=(CLUSTERHFHADID <<shiftId_) |clusref.key();
        }
      break;
      case reco::PFBlockElement::GSF:
	{
	  const reco::PFBlockElementGsfTrack *  GsfEl =  
	    dynamic_cast<const reco::PFBlockElementGsfTrack*>(&(*iter));
	  
	  reco::GsfPFRecTrackRef trackref=GsfEl->GsftrackRefPF();
	  assert( !trackref.isNull() ); 
	  ident=(GSFRECTRACKID << shiftId_) | trackref.key();
	}
      break;
      case reco::PFBlockElement::BREM:
	{
	  const reco::PFBlockElementBrem * Brem =  dynamic_cast<const reco::PFBlockElementBrem*>(&(*iter)); 
	  reco::GsfPFRecTrackRef trackref = Brem->GsftrackRefPF();
	  unsigned index=trackref.key()*40+Brem->indTrajPoint();
	  ident = (BREMID << shiftId_ ) | index ;
	  if (findBadBremsId(ident))  ident=-1;
	}
      break;

      case reco::PFBlockElement::SC:
	{
	  const reco::PFBlockElementSuperCluster * sc =  dynamic_cast<const reco::PFBlockElementSuperCluster*>(&(*iter)); 
	  reco::SuperClusterRef scref = sc->superClusterRef();
	  assert( !scref.isNull() ); 
	  ident = (CLUSTERIBID << shiftId_ ) | scref.key();
	}
      break;

      default: 
        std::cout<<"unknown PFBlock element of type "<<type<<std::endl;
        break; 
      } //end switch 
      pair <int, int> idElem;
      idElem.first=ident;
      idElem.second=(*iter).index();
      if (ident != -1) blockIdentsMap_.insert(pair<int,pair <int,int> > (ibl,idElem));            
    }   //end for elements
  }   //end for blocks
        
}
//__________________________________________________________________________________
void DisplayManager::loadGraphicObjects()
{
  loadGClusters();
  loadGRecHits();
  loadGRecTracks();
  loadGGsfRecTracks();
  loadGSimParticles();
  loadGPFBlocks();
  loadGGenParticles();
}
//________________________________________________________
void DisplayManager::loadGRecHits()
{
  double phi0=0;
        
  double maxee = getMaxEEcal();
  double maxeh = getMaxEHcal();
  double maxe = maxee>maxeh ? maxee : maxeh;
        
  int color = TColor::GetColor(210,210,210);
  int seedcolor = TColor::GetColor(145,145,145);
  int specialcolor = TColor::GetColor(255,140,0);
        
  for(unsigned i=0; i<em_->rechitsECAL_.size(); i++) { 
    int rhcolor = color;
    if( unsigned col = em_->clusterAlgoECAL_.color(i) ) {
      switch(col) {
      case PFClusterAlgo::SEED: rhcolor = seedcolor; break;
      case PFClusterAlgo::SPECIAL: rhcolor = specialcolor; break;
      default:
        cerr<<"DisplayManager::loadGRecHits: unknown color"<<endl;
      }
    }
    //int recHitId=(i<<shiftId_) | RECHITECALID;
    int recHitId=i;
    createGRecHit(em_->rechitsECAL_[i],recHitId, maxe, phi0, rhcolor);
  }
        
  for(unsigned i=0; i<em_->rechitsHCAL_.size(); i++) { 
    int rhcolor = color;
    if(unsigned col = em_->clusterAlgoHCAL_.color(i) ) {
      switch(col) {
      case PFClusterAlgo::SEED: rhcolor = seedcolor; break;
      case PFClusterAlgo::SPECIAL: rhcolor = specialcolor; break;
      default:
        cerr<<"DisplayManager::loadGRecHits: unknown color"<<endl;
      }
    }
    //int recHitId=(i<<shiftId_) | RECHITHCALID;
    int recHitId=(RECHITHCALID <<shiftId_) | i;
    createGRecHit(em_->rechitsHCAL_[i],recHitId, maxe, phi0, rhcolor);
  }
      
  if (drawHO_) {
    for(unsigned i=0; i<em_->rechitsHO_.size(); i++) { 
      int rhcolor = color;
      if(unsigned col = em_->clusterAlgoHO_.color(i) ) {
	switch(col) {
	case PFClusterAlgo::SEED: rhcolor = seedcolor; break;
	case PFClusterAlgo::SPECIAL: rhcolor = specialcolor; break;
	default:
	  cerr<<"DisplayManager::loadGRecHits: unknown color"<<endl;
	}
      }
      //int recHitId=(i<<shiftId_) | RECHITHOID;
      int recHitId=(RECHITHOID <<shiftId_) | i;
      createGRecHit(em_->rechitsHO_[i],recHitId, 2*maxe, phi0, rhcolor);
    }
  }
 
  for(unsigned i=0; i<em_->rechitsHFEM_.size(); i++) { 
    int rhcolor = color;
    if(unsigned col = em_->clusterAlgoHFEM_.color(i) ) {
      switch(col) {
      case PFClusterAlgo::SEED: rhcolor = seedcolor; break;
      case PFClusterAlgo::SPECIAL: rhcolor = specialcolor; break;
      default:
        cerr<<"DisplayManager::loadGRecHits: unknown color"<<endl;
      }
    }
    //int recHitId=(i<<shiftId_) | RECHITHFEMID;
    int recHitId=(RECHITHFEMID <<shiftId_) | i;
    createGRecHit(em_->rechitsHFEM_[i],recHitId, maxe, phi0, rhcolor);
  }
        
  for(unsigned i=0; i<em_->rechitsHFHAD_.size(); i++) { 
    int rhcolor = color;
    if(unsigned col = em_->clusterAlgoHFHAD_.color(i) ) {
      switch(col) {
      case PFClusterAlgo::SEED: rhcolor = seedcolor; break;
      case PFClusterAlgo::SPECIAL: rhcolor = specialcolor; break;
      default:
        cerr<<"DisplayManager::loadGRecHits: unknown color"<<endl;
      }
    }
    //int recHitId=(i<<shiftId_) | RECHITHFHADID;
    int recHitId=(RECHITHFHADID <<shiftId_) | i;
    createGRecHit(em_->rechitsHFHAD_[i],recHitId, maxe, phi0, rhcolor);
  }
        
  for(unsigned i=0; i<em_->rechitsPS_.size(); i++) { 
    int rhcolor = color;
    if( unsigned col = em_->clusterAlgoPS_.color(i) ) {
      switch(col) {
      case PFClusterAlgo::SEED: rhcolor = seedcolor; break;
      case PFClusterAlgo::SPECIAL: rhcolor = specialcolor; break;
      default:
        cerr<<"DisplayManager::loadGRecHits: unknown color"<<endl;
      }
    }
    //int recHitId=(i<<shiftId_) | RECHITPSID;
    int recHitId=(RECHITPSID<<shiftId_) | i;
                
    createGRecHit(em_->rechitsPS_[i],recHitId, maxe, phi0, rhcolor);
  }
} 
//________________________________________________________________________
void DisplayManager::loadGRecTracks()
{
  double phi0=0;
        
  int ind=-1;
  std::vector<reco::PFRecTrack>::iterator itRecTrack;
  for (itRecTrack = em_->recTracks_.begin(); itRecTrack != em_->recTracks_.end();itRecTrack++) {
    double sign = 1.;
    const reco::PFTrajectoryPoint& tpinitial 
      = itRecTrack->extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach);
    double pt = tpinitial.momentum().Pt();
    //if( pt<em_->displayRecTracksPtMin_ ) continue;
                
    const reco::PFTrajectoryPoint& tpatecal 
      = itRecTrack->trajectoryPoint(itRecTrack->nTrajectoryMeasurements() +
                                    reco::PFTrajectoryPoint::ECALEntrance );
                
    if ( cos(phi0 - tpatecal.momentum().Phi()) < 0.)
      sign = -1.;
                
    const std::vector<reco::PFTrajectoryPoint>& points = 
      itRecTrack->trajectoryPoints();
                
    int    linestyle = itRecTrack->algoType();
    ind++;
    //int recTrackId=(ind<<shiftId_) | RECTRACKID;
    int recTrackId=(RECTRACKID <<shiftId_) | ind; 
                
    createGTrack(*itRecTrack,points,recTrackId, pt, phi0, sign, false,linestyle);
  }
}

//________________________________________________________________________
void DisplayManager::loadGGsfRecTracks()
{
  double phi0=0;
        
  int ind=-1;
  int indbrem=-1;
  
  // allows not to draw Brems with no informations in PFBlocks
  retrieveBadBrems();
  
  std::vector<reco::GsfPFRecTrack>::iterator itRecTrack;
  for (itRecTrack = em_->gsfrecTracks_.begin(); itRecTrack != em_->gsfrecTracks_.end();itRecTrack++) {
    double sign = 1.;
    const reco::PFTrajectoryPoint& tpinitial 
      = itRecTrack->extrapolatedPoint(reco::PFTrajectoryPoint::ClosestApproach);
    double pt = tpinitial.momentum().Pt();
    //if( pt<em_->displayRecTracksPtMin_ ) continue;
                
    const reco::PFTrajectoryPoint& tpatecal 
      = itRecTrack->trajectoryPoint(itRecTrack->nTrajectoryMeasurements() +
                                    reco::PFTrajectoryPoint::ECALEntrance );
                
    if ( cos(phi0 - tpatecal.momentum().Phi()) < 0.)
      sign = -1.;
                
    const std::vector<reco::PFTrajectoryPoint>& points = 
      itRecTrack->trajectoryPoints();
                
    int    linestyle = itRecTrack->algoType();
    ind++;
    int recTrackId=(GSFRECTRACKID <<shiftId_) | ind; 
    
    createGTrack(*itRecTrack,points,recTrackId, pt, phi0, sign, false,linestyle,1);

    // now the Brems - bad to copy, but problems otherwise
    std::vector<reco::PFBrem> brems=itRecTrack->PFRecBrem();
    unsigned nbrems=brems.size();
    for(unsigned ibrem=0;ibrem<nbrems;++ibrem)
      {
	unsigned indTrajPoint=brems[ibrem].indTrajPoint();
	if(indTrajPoint==99) continue;
	double signBrem = 1. ; // this is not the charge
	int  linestyleBrem = brems[ibrem].algoType(); 
	indbrem++;
	// build the index by hand. Assume that there are less 40 Brems per GSF track (ultrasafe!)
	// make it from the GSF index and the brem index
	unsigned indexBrem= ind*40+brems[ibrem].indTrajPoint();
	int recTrackIdBrem=(BREMID << shiftId_ ) | indexBrem;
	
	//check if there is information on this brem in the PFBlock
	//before creating a graphic object
	if (!findBadBremsId(recTrackIdBrem)) {

	  // the vertex is not stored, need to make it by hand
	  std::vector<reco::PFTrajectoryPoint> pointsBrem;
	  //the first two trajectory points are dummy; copy them
	  pointsBrem.push_back(brems[ibrem].trajectoryPoints()[0]);
	  pointsBrem.push_back(brems[ibrem].trajectoryPoints()[1]);

	  // then get the vertex from the GSF track
	  pointsBrem.push_back(itRecTrack->trajectoryPoint(indTrajPoint));
	
	  unsigned ntp=brems[ibrem].trajectoryPoints().size();
	  for(unsigned itp=2;itp<ntp;++itp)
	   {
	    pointsBrem.push_back(brems[ibrem].trajectoryPoints()[itp]);
	   }

	  double deltaP=brems[ibrem].DeltaP();
	  const reco::PFTrajectoryPoint& tpatecalbrem
	    = brems[ibrem].trajectoryPoint(brems[ibrem].nTrajectoryMeasurements() +
					 reco::PFTrajectoryPoint::ECALEntrance );
	
	  if ( cos(phi0 - tpatecalbrem.momentum().Phi()) < 0.)
	    signBrem = -1.; // again, not the charge
	
	  createGTrack(brems[ibrem],pointsBrem,recTrackIdBrem,deltaP,phi0,signBrem,false,linestyleBrem,2);
	}  
      }
  }
 
}

//___________________________________________________________________________
void DisplayManager::loadGSimParticles()
{
  double phi0=0;
  //    bool debug_loadGSim = true;
  unsigned simParticlesVSize = em_->trueParticles_.size();
        
  for(unsigned i=0; i<simParticlesVSize; i++) {
                
    const reco::PFSimParticle& ptc = em_->trueParticles_[i];
                
    const reco::PFTrajectoryPoint& tpinitial 
      = ptc.extrapolatedPoint( reco::PFTrajectoryPoint::ClosestApproach );
                
    double pt = tpinitial.momentum().Pt();
    //if( pt<em_->getDisplayTrueParticlesPtMin()) continue;
                
    double sign = 1.;
    const reco::PFTrajectoryPoint& tpFirst = ptc.trajectoryPoint(0);
    if ( tpFirst.position().X() < 0. )
      sign = -1.;
    //double sign2 = 1.;
    // if position vector is 0,0,0: X component is undefined  (MDN 11 June 2008)
    // const reco::PFTrajectoryPoint& tpFirst = ptc.trajectoryPoint(0);
    //  if ( tpFirst.positionXYZ().X() < 0. )
    //   sign2 = -1.;

    const std::vector<reco::PFTrajectoryPoint>& points = 
      ptc.trajectoryPoints();
      

    int indexMarker;
    switch( std::abs(ptc.pdgCode() ) ) {
    case 22:   indexMarker=0; break; // photons
    case 11:   indexMarker=1;  break; // electrons 
    case 13:   indexMarker=2;  break; // muons 
    case 130:  
    case 321:  indexMarker=3; break; // K
    case 211:  indexMarker=4; break; // pi+/pi-
    case 2212: indexMarker=5; break; // protons
    case 2112: indexMarker=6; break; // neutrons  
    default:   indexMarker=7; break; 
    }

    bool displayInitial=true;
    if( ptc.motherId() < 0 ) displayInitial=false;
    int partId=(SIMPARTICLEID << shiftId_) | i; 
    createGPart(ptc, points,partId, pt, phi0, sign, displayInitial,indexMarker);
    //cout << " sign " << sign << " sign2 " << sign2 << endl;
    //if ( sign*sign2 <0 ) cout << " ++++++Warning sign*sign2 <0 ++++++++++++++++++ " << endl;
                
  }
}
//_____________________________________________________________________________
void DisplayManager::lookForMaxRecHit(bool ecal)
{
  // look for the rechit with max e in ecal or hcal
  double maxe = -999;
  reco::PFRecHit* maxrh = 0;
        
  reco::PFRecHitCollection* rechits = 0;
  if(ecal) rechits = &(em_->rechitsECAL_);
  else rechits = &(em_->rechitsHCAL_);
  assert(rechits);
        
  for(unsigned i=0; i<(*rechits).size(); i++) {
                
    double energy = (*rechits)[i].energy();
                
    if(energy > maxe ) {
      maxe = energy;
      maxrh = &((*rechits)[i]);
    }      
  }
        
  if(!maxrh) return;
        
  // center view on this rechit
        
        
  // get the cell size to set the eta and phi width 
  // of the display window from one of the cells
        
  double phisize = -1;
  double etasize = -1;
  maxrh->size(phisize, etasize);
        
  double etagate = zoomFactor_ * etasize;
  double phigate = zoomFactor_ * phisize;
  
  double eta =  maxrh->position().Eta();
  double phi =  maxrh->position().Phi();
  
  if(displayHist_[EPE]) {
    displayHist_[EPE]->GetXaxis()->SetRangeUser(eta-etagate, eta+etagate);
    displayHist_[EPE]->GetYaxis()->SetRangeUser(phi-phigate, phi+phigate);
    displayView_[EPE]->Modified();
    displayView_[EPE]->Update();
  }
        
  if(displayHist_[EPH]) {
    displayHist_[EPH]->GetXaxis()->SetRangeUser(eta-etagate, eta+etagate);
    displayHist_[EPH]->GetYaxis()->SetRangeUser(phi-phigate, phi+phigate);
    displayView_[EPH]->Modified();
    displayView_[EPH]->Update();
  }

  if(drawHO_ && displayHist_[EHO]) {
    displayHist_[EHO]->GetXaxis()->SetRangeUser(eta-etagate, eta+etagate);
    displayHist_[EHO]->GetYaxis()->SetRangeUser(phi-phigate, phi+phigate);
    displayView_[EHO]->Modified();
    displayView_[EHO]->Update();
  }


}
//________________________________________________________________________________
void DisplayManager::lookForGenParticle(unsigned barcode) {
        
  const HepMC::GenEvent* event = em_->MCTruth_.GetEvent();
  if(!event) {
    cerr<<"no GenEvent"<<endl;
    return;
  }
        
  const HepMC::GenParticle* particle = event->barcode_to_particle(barcode);
  if(!particle) {
    cerr<<"no particle with barcode "<<barcode<<endl;
    return;
  }
        
  math::XYZTLorentzVector momentum(particle->momentum().px(),
                                   particle->momentum().py(),
                                   particle->momentum().pz(),
                                   particle->momentum().e());
        
  double eta = momentum.Eta();
  double phi = momentum.phi();
        
  double phisize = 0.05;
  double etasize = 0.05;
        
  double etagate = zoomFactor_ * etasize;
  double phigate = zoomFactor_ * phisize;
        
  if(displayHist_[EPE]) {
    displayHist_[EPE]->GetXaxis()->SetRangeUser(eta-etagate, eta+etagate);
    displayHist_[EPE]->GetYaxis()->SetRangeUser(phi-phigate, phi+phigate);
    displayView_[EPE]->Modified();
    displayView_[EPE]->Update();
                
  }
  if(displayHist_[EPH]) {
    displayHist_[EPH]->GetXaxis()->SetRangeUser(eta-etagate, eta+etagate);
    displayHist_[EPH]->GetYaxis()->SetRangeUser(phi-phigate, phi+phigate);
    displayView_[EPH]->Modified();
    displayView_[EPH]->Update();
  }

  if(drawHO_ && displayHist_[EHO]) {
    displayHist_[EHO]->GetXaxis()->SetRangeUser(eta-etagate, eta+etagate);
    displayHist_[EHO]->GetYaxis()->SetRangeUser(phi-phigate, phi+phigate);
    displayView_[EHO]->Modified();
    displayView_[EHO]->Update();
  }

}
//_______________________________________________________________________
void DisplayManager::printDisplay(const char* sdirectory ) const
{
  string directory = sdirectory;
  if( directory.empty() ) {   
    directory = "Event_";
  }
  char num[10];
  sprintf(num,"%d", eventNumber_);
  directory += num;
        
  string mkdir = "mkdir "; mkdir += directory;
  int code = system( mkdir.c_str() );
        
  if( code ) {
    cerr<<"cannot create directory "<<directory<<endl;
    return;
  }
        
  cout<<"Event display printed in directory "<<directory<<endl;
        
  directory += "/";
        
  for(unsigned iView=0; iView<displayView_.size(); iView++) {
    if( !displayView_[iView] ) continue;
                
    string name = directory;
    name += displayView_[iView]->GetName();
                
    cout<<displayView_[iView]->GetName()<<endl;
                
    string eps = name; eps += ".eps";
    displayView_[iView]->SaveAs( eps.c_str() );
                
    string png = name; png += ".png";
    displayView_[iView]->SaveAs( png.c_str() );
  }
        
  string txt = directory;
  txt += "event.txt";
  ofstream out( txt.c_str() );
  if( !out ) 
    cerr<<"cannot open "<<txt<<endl;
  em_->print( out );
}
//_____________________________________________________________________________
void DisplayManager::reset()
{
  maxERecHitEcal_=-1;
  maxERecHitHcal_=-1;
  maxERecHitHo_=-1;
  isGraphicLoaded_= false;
  
  std::multimap<int,GPFBase *>::iterator p;
  for (p=graphicMap_.begin();p!=graphicMap_.end();p++)
    delete p->second;
  graphicMap_.clear();

  blockIdentsMap_.clear(); 
  selectedGObj_.clear();
  badBremsId_.clear();
  

}
//_______________________________________________________________________________
void DisplayManager::unZoom()
{
  for( unsigned i=0; i<displayHist_.size(); i++) {
    // the corresponding view was not requested
    if( ! displayHist_[i] ) continue;
    displayHist_[i]->GetXaxis()->UnZoom();
    displayHist_[i]->GetYaxis()->UnZoom();
  }
  updateDisplay();
}
//_______________________________________________________________________________
void DisplayManager::setNewAttrToSimParticles()
{
  simPartPatternM_.clear();
  simPartPatternM_.push_back(simPartPatternPhoton_);
  simPartPatternM_.push_back(simPartPatternElec_);
  simPartPatternM_.push_back(simPartPatternMuon_);
  simPartPatternM_.push_back(simPartPatternK_);
  simPartPatternM_.push_back(simPartPatternPi_);
  simPartPatternM_.push_back(simPartPatternProton_);
  simPartPatternM_.push_back(simPartPatternNeutron_);
  simPartPatternM_.push_back(simPartPatternDefault_);
} 

//_______________________________________________________________________________
void DisplayManager::printGenParticleInfo(std::string name,int barcode,int barcodeMother) 
{ 
  const HepMC::GenEvent* myGenEvent = em_->MCTruth_.GetEvent();
  HepMC::GenParticle *p = myGenEvent->barcode_to_particle(barcode);
  std::cout<<"genParticle "<<name<<" with barcode "<<barcode<<std::flush<<std::endl;
  p->print();
  if (barcodeMother) { 
    HepMC:: GenParticle *mother = myGenEvent->barcode_to_particle(barcodeMother);
    std::cout<<"mother particle with barcode "<<barcodeMother<<std::flush<<std::endl;
    mother->print();
  }    
}
