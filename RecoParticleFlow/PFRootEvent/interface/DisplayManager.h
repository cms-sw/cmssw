#ifndef PF_DisplayManager_h
#define PF_DisplayManager_h

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayCommon.h"

#include <TCanvas.h>
#include <TLine.h>
#include <TBox.h>
#include <string>

class GPFRecHit;
class GPFCluster;
class GPFTrack;
class GPFSimParticle;

class DisplayManager {
  
   public:
     DisplayManager(PFRootEventManager *em);
     virtual ~DisplayManager();
     
     void display(int ientry);
     void displayAll();
     void displayNext();
     void displayNextInteresting(int ientry);
     void displayPrevious();
     /// look for particle with index i in MC truth.
     void lookForGenParticle(unsigned barcode);
     /// look for rechit with max energy in ecal or hcal.
     void lookForMaxRecHit(bool ecal);
     void reset();
     void updateDisplay();
     void unZoom();
     void printDisplay(const char* directory="" ) const;
     //bool getGLoaded() {return isGraphicLoaded_;}  //util to DialogFrame ?
     
     //variables
     //-------------------------------graphic options variable ---------------------
     double clusEnMin_;
     double hitEnMin_;
     double trackPtMin_;
     double particlePtMin_;
     
     bool drawHits_;
     bool drawTracks_;
     bool drawClus_;
     bool drawClusterL_;
     bool drawParticles_;
     
   private:
   
     PFRootEventManager *em_;
     double maxERecHitEcal_;
     double maxERecHitHcal_;
     bool   isGraphicLoaded_;
     int    eventNb_;
     int    maxEvents_;
     double zoomFactor_;

    //-------------- draw Canvas --------------------------------------
    /// vector of canvas for x/y or r/z display
    std::vector<TCanvas*> displayView_;
    /// display pad xy size for (x,y) or (r,z) display
    std::vector<int>      viewSize_; 
        
    /// display pad xy size for eta/phi view
    std::vector<int>         viewSizeEtaPhi_; 

    /// support histogram for x/y or r/z display. 
    std::vector<TH2F*>    displayHist_;

     /// ECAL in XY view. \todo should be attribute ?
    TEllipse frontFaceECALXY_;

    /// ECAL in RZ view. \todo should be attribute ?
    TBox     frontFaceECALRZ_;

     /// HCAL in XY view. \todo should be attribute ?
    TEllipse frontFaceHCALXY_;
    //----------------  end Draw Canvas ------------------------------------
    
     /// graphic object containers     
     std::vector< std::vector<GPFRecHit> >       vectGHits_;
     std::vector< std::vector<GPFCluster> >      vectGClus_;
     std::vector< std::vector<GPFTrack> >        vectGTracks_;
     std::vector< std::vector<GPFSimParticle> >  vectGParts_;
     std::vector<std::vector<TLine> >            vectGClusterLines_;
     //number of clusterLines by cluster 
     std::vector<std::vector<int> >              vectClusLNb_;
     
     
    
     // Display Options read from option file
     void  getDisplayOptions();
     
     //DisplayViews
     void createCanvas();
     void displayCanvas();
     
     
     //create graphicObjects
     void createGRecHit(reco::PFRecHit& rh, double maxe, double phi0=0. , int color=4);
     void createGCluster(const reco::PFCluster& cluster,double phi0 = 0.);
     void createGTrack(reco::PFRecTrack &tr,const std::vector<reco::PFTrajectoryPoint>& points, 
 		       double pt,double phi0, double sign, bool displayInitial, int linestyle);
     void createGPart(const reco::PFSimParticle &ptc, const std::vector<reco::PFTrajectoryPoint>& points, 
 		      double pt,double phi0, double sign, bool displayInitial, int markerstyle);
     void createGClusterLines(const reco::PFCluster& cluster,int viewType);		      

     //fill vectors with graphic Objects
     void loadGraphicObjects();
     void loadGRecHits();
     void loadGClusters();
     void loadGRecTracks();
     void loadGTrueParticles();
     
     /// draw methods
     void drawRecHits(int view,double enmin);
     void drawClusters(int view,double enmin);
     void drawTracks(int view,double ptmin);
     void drawParts(int view,double ptmin);
     void drawClusterLines(int clusIndex,int viewType,int &istart);
     

     // methods
     double getMaxE(int layer) const;
     double getMaxEEcal();
     double getMaxEHcal();
     	       
     
     
     
};
#endif 
