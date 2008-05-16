#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "RecoParticleFlow/PFRootEvent/interface/IO.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "RecoParticleFlow/PFBlockAlgo/interface/PFGeometry.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/Math/interface/Point3D.h"


#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFRecHit.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFTrack.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFSimParticle.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFGenParticle.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h"

#include <TTree.h>
#include <TVector3.h>
#include <TH2F.h>
#include <TEllipse.h>
#include "TLine.h"
#include "TLatex.h"
#include "TList.h"
#include "TColor.h"
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
  isGraphicLoaded_(false),
  shiftId_(27) {
  
  readOptions( optfile );
  
  eventNumber_  = em_->eventNumber();
  //TODOLIST: re_initialize if new option file- better in em
  maxEvents_= em_->tree_->GetEntries();  
  
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
  if(clusterAttributes_.size() != 4) {
    cerr<<"PFRootEventManager::::ReadOptions, bad display/cluster_attributes tag...using 20 10 2 5"
        <<endl;
    clusterAttributes_.clear();
    clusterAttributes_.push_back(2); //color
    clusterAttributes_.push_back(5);  // color if clusterPS
    clusterAttributes_.push_back(20); // marker style
    clusterAttributes_.push_back(10); //markersize *10
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
    trackAttributes_.push_back(8);   //Marker size *10
  }
  
  
  clusPattern_ = new TAttMarker(clusterAttributes_[0],clusterAttributes_[2],(double)clusterAttributes_[3]/10);
  clusPSPattern_ = new TAttMarker(clusterAttributes_[1],clusterAttributes_[2],(double)clusterAttributes_[3]/10);
  trackPatternL_ = new TAttLine(trackAttributes_[0],trackAttributes_[1],1);
  trackPatternM_ = new TAttMarker(trackAttributes_[0],trackAttributes_[2],(double)trackAttributes_[3]/10);
  
  genPartPattern_= new TAttMarker(kGreen-1,22,1.);
  
  simPartPatternPhoton_ = new TAttMarker(4,3,.8);
  simPartPatternElec_   = new TAttMarker(4,5,.8);
  simPartPatternMuon_   = new TAttMarker(4,2,.8);
  simPartPatternK_      = new TAttMarker(4,24,.8);
  simPartPatternPi_     = new TAttMarker(4,25,.8);
  simPartPatternProton_ = new TAttMarker(4,26,.8);
  simPartPatternNeutron_= new TAttMarker(4,27,.8);
  simPartPatternDefault_= new TAttMarker(4,30,.8);
  
  simPartPatternL_ = new TAttLine(4,2,1);
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

  drawParticles_ = true;
  options_->GetOpt("display", "particles", drawParticles_);

  particlePtMin_ = -1;
  options_->GetOpt("display", "particles_ptmin", particlePtMin_);
  
  
  drawGenParticles_=false;
  genParticlePtMin_ = 0;
  
  
  trackPtMin_ = -1;
  options_->GetOpt("display", "rectracks_ptmin", trackPtMin_);
  
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
  
  
  for (int viewType=0;viewType<NViews;++viewType) {
    displayView_[viewType]->SetGrid(0, 0);
    displayView_[viewType]->SetLeftMargin(0.12);
    displayView_[viewType]->SetBottomMargin(0.12);
    displayView_[viewType]->ToggleToolBar();
  } 
    
  // Draw support histogram
  double zLow = -500.;
  double zUp  = +500.;
  double rLow = -300.;
  double rUp  = +300.;
  displayHist_[XY] = new TH2F("hdisplayHist_XY", "", 500, rLow, rUp, 
                              500, rLow, rUp);
  displayHist_[XY]->SetXTitle("X");
  displayHist_[XY]->SetYTitle("Y");
  
  displayHist_[RZ] = new TH2F("hdisplayHist_RZ", "",500, zLow, zUp, 
                              500, rLow, rUp); 
  displayHist_[RZ]->SetXTitle("Z");
  displayHist_[RZ]->SetYTitle("R");
  
  displayHist_[EPE] = new TH2F("hdisplayHist_EP", "", 500, -5, 5, 
                               500, -3.5, 3.5);
  displayHist_[EPE]->SetXTitle("#eta");
  displayHist_[EPE]->SetYTitle("#phi");
  
  displayHist_[EPH] = displayHist_[EPE];
  
  for (int viewType=0;viewType<NViews;++viewType){
    displayHist_[viewType]->SetStats(kFALSE);
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
  
  // Draw ECAL side
  frontFaceECALRZ_.SetX1(-1.*PFGeometry::innerZ(PFGeometry::ECALEndcap));
  frontFaceECALRZ_.SetY1(-1.*PFGeometry::innerRadius(PFGeometry::ECALBarrel));
  frontFaceECALRZ_.SetX2(PFGeometry::innerZ(PFGeometry::ECALEndcap));
  frontFaceECALRZ_.SetY2(PFGeometry::innerRadius(PFGeometry::ECALBarrel));
  frontFaceECALRZ_.SetFillStyle(0);

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

  const math::XYZPoint& xyzPos = cluster.position();
  GPFCluster *gc;
  
  for (int viewType=0;viewType<4;viewType++){

    switch(viewType) {
    case XY:
      {
       if (clusType==0) {
          gc = new  GPFCluster(this,
                                       viewType,ident,
                                       &cluster,
                                       xyzPos.X(), xyzPos.Y(), clusPattern_);
       }
       else {
          gc = new  GPFCluster(this,
                                       viewType,ident,
                                       &cluster,
                                      xyzPos.X(), xyzPos.Y(), clusPSPattern_);
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
	}
	else {
	  gc = new  GPFCluster(this,
	                                 viewType,ident,
					 &cluster,
					 xyzPos.z(),sign*xyzPos.Rho(),
					 clusPattern_);
	}
					 
	graphicMap_.insert(pair<int,GPFBase *>	(ident, gc));			 
      } 
      break;
    case EPE:
      {
        if( cluster.layer()<0 ) {
	  if (clusType==0) {
	     gc = new  GPFCluster(this,
	                                 viewType,ident,
					 &cluster,
					 eta,phi,
					 clusPattern_);
	  }
	  else {
	     gc = new  GPFCluster(this,
	                                 viewType,ident,
					 &cluster,
					 eta,phi,
					 clusPSPattern_);
	  }				 
					 
	graphicMap_.insert(pair<int,GPFBase *>	(ident, gc));			 
       }
      } 
      break;
    case EPH:
      {
        if( cluster.layer()>0 ) {
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
	    
					 
	  graphicMap_.insert(pair<int,GPFBase *>	(ident, gc));			 
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
  
  for (int viewType=0;viewType<4;++viewType) {
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
        xPos.push_back( eta );
        yPos.push_back( phi );
        break;
      default:break;
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
					      
    graphicMap_.insert(pair<int,GPFBase *>	(ident, gc));				      
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
  case PFLayer::HCAL_HF: 
    // how to handle a different threshold for HF?
    thresh = em_->clusterAlgoHCAL_.threshEndcap();
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
  for(int viewType=0;viewType<4;++viewType) {
  
    bool isHCAL = (layer == PFLayer::HCAL_BARREL1 || 
		   layer == PFLayer::HCAL_BARREL2 || 
		   layer == PFLayer::HCAL_ENDCAP || 
		   layer == PFLayer::HCAL_HF);
    
    if(  viewType == EPH && 
         ! isHCAL) {
      continue;
    }
    // on EPE view, draw only HCAL and preshower
    if(  viewType == EPE && isHCAL ) {
      continue;
    }
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
            ( viewType == XY &&  
              ( layer == PFLayer::ECAL_ENDCAP || 
                layer == PFLayer::HCAL_ENDCAP || 
		layer == PFLayer::HCAL_HF
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
        graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,eta,phi,color,"l")));
     
        if( ampl>0 ) { // not for preshower
          etaprop[4]=etaprop[0];
          phiprop[4]=phiprop[0]; // closing the polycell    
          graphicMap_.insert(pair<int,GPFBase *> (ident,new GPFRecHit(this, viewType,ident,&rh,npoints,etaprop,phiprop,color,"f")));
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
                                   int linestyle) 
{
      
  //   bool inside = false; 
  //TCutG* cutg = (TCutG*) gROOT->FindObject("CUTG");
  
  for (int viewType=0;viewType<4;++viewType) {
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
        xPos.push_back( eta );
        yPos.push_back( phi );
        break;
      }
    }  
    /// no point inside graphical cut.
    //if( !inside ) return;
  
    //fill map with graphic objects
     
    GPFTrack *gt = new  GPFTrack(this,
                                 viewType,ident,
                                 &tr,
                                 xPos.size(),&xPos[0],&yPos[0],pt,
				 trackPatternM_,trackPatternL_,"pl");
    graphicMap_.insert(pair<int,GPFBase *> (ident, gt));
  }   
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
    switch (type) {
    case CLUSTERECALID: case CLUSTERHCALID: case  CLUSTERPSID: case CLUSTERIBID:
      {
        if (drawClus_)
          if (p->second->getEnergy() > clusEnMin_) {
            displayView_[view]->cd();
            p->second->draw();
          }        
      }
      break;
    case RECHITECALID: case  RECHITHCALID: case RECHITPSID:
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
	    if (view == EPH || view ==EPE) {
              displayView_[view]->cd();
              p->second->draw();
	    }  
      } 
      break;	       
    default : std::cout<<"DisplayManager::displayAll()-- unknown object "<<std::endl;               
    }  //switch end
  }   //for end
  for (int i=0;i<NViews;i++) {
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
    switch (type) {
    case CLUSTERECALID: case CLUSTERHCALID: case  CLUSTERPSID: case CLUSTERIBID:
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
  double rUp  = +300.;
 
  //TODOLIST : test wether view on/off
  //if (!displayView_[viewType] || !gROOT->GetListOfCanvases()->FindObject(displayView_[viewType]) ) {
  //   assert(viewSize_.size() == 2);
  
  for (int viewType=0;viewType<NViews;++viewType) {
    displayView_[viewType]->cd();
    displayHist_[viewType]->Draw();
    switch(viewType) {
    case XY: 
      frontFaceECALXY_.Draw();
      frontFaceHCALXY_.Draw();
      break;
    case RZ:    
      {// Draw lines at different etas
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
  while (!ok && ientry<em_->tree_->GetEntries() ) {
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
    return;
  }
  p=result.first;
  while (p != result.second) {
    int view=p->second->getView();
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
//_______________________________________________________________________________
void DisplayManager::findAndDraw(int ident) 
{

  int type=ident >> shiftId_;
  int color=1;
  if (type>8) {
    std ::cout<<"DisplayManager::findAndDraw :object Type unknown"<<std::endl;
    return;
  }  
  if (drawPFBlocks_==0  || type<3 || type==8) {
    rubOutGPFBlock();
    selectedGObj_.clear();
    bool toInitial=false;
    drawGObject(ident,color,toInitial);
    if (type<3) {
      //redrawWithoutHits_=true;
      displayAll(false);
      //redrawWithoutHits_=false;
    }
  }     
  updateDisplay();
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
//_________________________________________________________________________

void DisplayManager::updateDisplay() {
  for(unsigned i=0; i<displayView_.size(); i++) {
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
  case PFLayer::HCAL_BARREL2:
  case PFLayer::HCAL_HF:
    vec = &(em_->rechitsHCAL_);
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
    maxERecHitEcal_ = maxeec > maxeb ? maxeec:maxeb; 
    // max of both barrel and endcap
  }
  return  maxERecHitEcal_;
}
//_______________________________________________________________________________
double DisplayManager::getMaxEHcal() {

  if(maxERecHitHcal_ < 0) {
    double maxehf = getMaxE( PFLayer::HCAL_HF );
    double maxeec = getMaxE( PFLayer::HCAL_ENDCAP );
    double maxeb =  getMaxE( PFLayer::HCAL_BARREL1 );
    maxERecHitHcal_ =  maxeec>maxeb  ?  maxeec:maxeb;
    maxERecHitHcal_ = maxERecHitHcal_>maxehf ? maxERecHitHcal_:maxehf;
  }
  return maxERecHitHcal_;
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
      if ( !p->production_vertex() ) continue;
      createGGenParticle(p);
 } 
}
//____________________________________________________________________________
void DisplayManager::createGGenParticle(HepMC::GenParticle* p)
{
    
    int partId = p->pdg_id();
    std::string name;
    std::string latexStringName;

    name = em_->getGenParticleName(partId,latexStringName);
    int barcode = p->barcode();
    int genPartId=(GENPARTICLEID<<shiftId_) | barcode;
    
    
    int vertexId1 = 0;
    vertexId1 = p->production_vertex()->barcode();
    
    math::XYZVector vertex1 (p->production_vertex()->position().x()/10.,
                             p->production_vertex()->position().y()/10.,
                             p->production_vertex()->position().z()/10.);
			     
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
    if( p->production_vertex()->particles_in_size() ) {
      mother = 
	*(p->production_vertex()->particles_in_const_begin()); 
    }

    // Colin: no need to declare this pointer in this context.
    // the declaration can be more local.
    // GPFGenParticle *gp; 

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
       
       for (int view = 2; view< NViews; view++) {
         GPFGenParticle* gp   = new GPFGenParticle(this,             
						   view, genPartId,
						   x, y,              //double *, double *
						   e,pt,barcode,barcodeMother,
						   genPartPattern_, 
						   name,latexStringName);
	 graphicMap_.insert(pair<int,GPFBase *>	(genPartId, gp));
       }
    }
    else {     //no Mother    
      for (int view = 2; view< NViews; view++) {
        GPFGenParticle* gp   = new GPFGenParticle(this,
						  view, genPartId,
						  eta, phi,                  //double double
						  e,pt,barcode,
						  genPartPattern_,
						  name, latexStringName);
        graphicMap_.insert(pair<int,GPFBase *>	(genPartId, gp));
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
  for(unsigned i=0; i<em_->clustersPS_->size(); i++){ 
    //int clusId=(i<<shiftId_) | CLUSTERPSID;
    int clusId=(CLUSTERPSID<<shiftId_) | i;
    createGCluster( (*(em_->clustersPS_))[i],clusId,phi0);
  }
  for(unsigned i=0; i<em_->clustersIslandBarrel_.size(); i++) {
    PFLayer::Layer layer = PFLayer::ECAL_BARREL;
    //int clusId=(i<<shiftId_) | CLUSTERIBID;
    int clusId=(CLUSTERIBID<<shiftId_) | i;
   
    reco::PFCluster cluster( layer, 
                             em_->clustersIslandBarrel_[i].energy(),
                             em_->clustersIslandBarrel_[i].x(),
                             em_->clustersIslandBarrel_[i].y(),
                             em_->clustersIslandBarrel_[i].z() ); 
    createGCluster( cluster,clusId, phi0);
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
         //std::cout<<"elem index "<<(*iter).index()<<"-type:"
         //      <<(*iter).type()<<std::flush<<std::endl;
      int ident=-1;  
       
      reco::PFBlockElement::Type type = (*iter).type();
      switch (type) {
      case reco::PFBlockElement::NONE :
        std::cout<<"unknown PFBlock element"<<std::endl;
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
      default: 
        std::cout<<"unknown PFBlock element"<<std::endl;
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
//___________________________________________________________________________
void DisplayManager::loadGSimParticles()
{
  double phi0=0;
  
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

    const std::vector<reco::PFTrajectoryPoint>& points = 
      ptc.trajectoryPoints();
      

   int markerstyle;
   int indexMarker;
    switch( abs(ptc.pdgCode() ) ) {
      case 22:   markerstyle = 3 ; indexMarker=0; break; // photons
      case 11:   markerstyle = 5 ; indexMarker=1;  break; // electrons 
      case 13:   markerstyle = 2 ; indexMarker=2;  break; // muons 
      case 130:  
      case 321:  markerstyle = 24; indexMarker=3; break; // K
      case 211:  markerstyle = 25; indexMarker=4; break; // pi+/pi-
      case 2212: markerstyle = 26; indexMarker=5; break; // protons
      case 2112: markerstyle = 27; indexMarker=6; break; // neutrons  
      default:   markerstyle = 30; indexMarker=7; break; 
    }
   
   
    bool displayInitial=true;
    if( ptc.motherId() < 0 ) displayInitial=false;
    
    int partId=(SIMPARTICLEID << shiftId_) | i; 
    createGPart(ptc, points,partId, pt, phi0, sign, displayInitial,indexMarker);
    
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
  isGraphicLoaded_= false;
  
  std::multimap<int,GPFBase *>::iterator p;
  for (p=graphicMap_.begin();p!=graphicMap_.end();p++)
    delete p->second;
  graphicMap_.clear();
 
  blockIdentsMap_.clear(); 
  selectedGObj_.clear();
  
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
