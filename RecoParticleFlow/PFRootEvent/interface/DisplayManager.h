#ifndef PF_DisplayManager_h
#define PF_DisplayManager_h

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayCommon.h"
#include "DataFormats/ParticleFlowReco/interface/PFBrem.h"

#include <TCanvas.h>
#include <TObject.h>
#include <TLine.h>
#include <TBox.h>
#include <string>
#include <map>

class IO;

class GPFRecHit;
class GPFCluster;
class GPFTrack;
class GPFSimParticle;
class GPFBase;
class GPFGenParticle;

class DisplayManager {
  
 public:
  DisplayManager( PFRootEventManager *em, 
                  const char* optfile );
  virtual ~DisplayManager();
     
  void readOptions( const char* file );

  void display(int ientry);
  void displayAll(bool noRedraw = true);
  void displayNext();
  void displayNextInteresting(int ientry);
  void displayPrevious();
  void displayPFBlock(int blockNb) ;
  void enableDrawPFBlock(bool state);
  void enableDrawBrem(bool state);
  void findAndDraw(int ident);
  //void findAndDrawbis(const int ident);
  void findBlock(int ident) ;
  /// look for particle with index i in MC truth.
  void lookForGenParticle(unsigned barcode);
  /// look for rechit with max energy in ecal or hcal.
  void lookForMaxRecHit(bool ecal);
  void reset();
  void updateDisplay();
  void unZoom();
  void printDisplay(const char* directory="" ) const;
  void printGenParticleInfo(std::string name, int barcode, int barcodeMother);
  void drawWithNewGraphicAttributes();
  void setNewAttrToSimParticles();
     
  //bool getGLoaded() {return isGraphicLoaded_;}  //util to DialogFrame ?
     
  //variables
  //----------------------graphic options variable ---------------------
  double clusEnMin_;
  double hitEnMin_;
  double trackPtMin_;
  double gsfPtMin_;
  double particlePtMin_;
  double genParticlePtMin_;
     
  bool drawHits_;
  bool drawTracks_;
  bool drawGsfTracks_;
  bool drawBrems_;
  bool drawClus_;
  bool drawClusterL_;
  bool drawParticles_;
  bool drawGenParticles_;
  bool drawPFBlocks_;
     
     
  //---------------------- new graphic Container ----------------
  //container of all the graphic Objects of one event 
  std::multimap<int,GPFBase *>  graphicMap_;
  //container of idents/elementNb  of objects within a PFBlock
  std::multimap<int , std::pair <int,int> >   blockIdentsMap_;
  
  //------------- graphic attributes ------------------------------------
  std::vector<float>      trackAttributes_;
  std::vector<float>      gsfAttributes_;
  std::vector<float>      bremAttributes_;
  std::vector<float>      clusterAttributes_;
  
  
  TAttMarker *clusPattern_;
  TAttMarker *clusPSPattern_;
  
  TAttMarker *trackPatternM_;
  TAttLine   *trackPatternL_;
  TAttMarker *gsfPatternM_;
  TAttLine   *gsfPatternL_;
  TAttMarker *bremPatternM_;
  TAttLine   *bremPatternL_;
  TAttMarker *genPartPattern_;
  
  TAttLine   *simPartPatternL_;
  TAttMarker *simPartPatternPhoton_;
  TAttMarker *simPartPatternElec_ ;
  TAttMarker *simPartPatternMuon_;
  TAttMarker *simPartPatternK_;
  TAttMarker *simPartPatternPi_;
  TAttMarker *simPartPatternProton_;
  TAttMarker *simPartPatternNeutron_;
  TAttMarker *simPartPatternDefault_;
  
  std::vector<TAttMarker *> simPartPatternM_;
  
  
 private:

  PFRootEventManager *em_;
   
  /// options file parser 
  IO*         options_;      

  double maxERecHitEcal_;
  double maxERecHitHcal_;
  bool   isGraphicLoaded_;
  int    eventNumber_;
  int    maxEvents_;
  double zoomFactor_;
  
  /// shiftId_ is the number of low bits indicating the object nb in the map key
  /// the object type is obtained by shifting the key number of shiftId_ bits
  /// ShiftId_ == SHIFTID == 26
  int    shiftId_;

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
  //std::vector<std::vector<TLine> >            vectGClusterLines_;
  //number of clusterLines by cluster 
  //std::vector<std::vector<int> >              vectClusLNb_;
    
  std::vector<int>                            selectedGObj_;
  std::vector<int>                            badBremsId_;
     
    
  // Display Options read from option file
  void  getDisplayOptions();
     
  //DisplayViews
  void createCanvas();

  void displayCanvas();
     
     
  //create graphicObjects
  void createGRecHit(reco::PFRecHit& rh,int ident, 
                     double maxe, double phi0=0. , int color=4);

  void createGCluster(const reco::PFCluster& cluster,
                      int ident, double phi0 = 0.);

  void createGTrack(reco::PFRecTrack &tr,
                    const std::vector<reco::PFTrajectoryPoint>& points,
                    int ident,double pt,double phi0, double sign, 
                    bool displayInitial, int linestyle, int kfgsfbrem=0);

  void createGPart(const reco::PFSimParticle &ptc, 
                   const std::vector<reco::PFTrajectoryPoint>& points, 
                   int ident,double pt,double phi0, double sign, 
                   bool displayInitial,int markerIndex);
                   
  //  void createGGenParticle(HepMC::GenEvent::particle_const_iterator p);                 
  void createGGenParticle(HepMC::GenParticle* p);                  

  //void createGClusterLines(const reco::PFCluster& cluster,int viewType);                      
  void drawGObject(int ident,int color,bool toInitialColor);

  //fill vectors with graphic Objects
  void loadGraphicObjects();

  void loadGRecHits();

  void loadGClusters();

  void loadGRecTracks();

  void loadGGsfRecTracks();

  void loadGSimParticles();

  void loadGPFBlocks();
  
  void loadGGenParticles();
     
  //void redraw();
  void rubOutGPFBlock();
  
  // 
  void retrieveBadBrems();
  bool findBadBremsId(int ident);
   

 
  // methods
  double getMaxE(int layer) const;
  double getMaxEEcal();
  double getMaxEHcal();
               
     
     
     
};
#endif 
