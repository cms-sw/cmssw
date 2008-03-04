#ifndef Graphic_PFCluster_h
#define Graphic_PFCluster_h

/*! \file interface/GPFCluster.h
  class to create graphic object
  from physical object PFCluster
*/  
 
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h" 
#include "TMarker.h"


class GPFCluster : public GPFBase, public TMarker {
 public:

  GPFCluster(DisplayManager *dm, int view, int ident, 
	     const reco::PFCluster* clus,
             double x,double y,int color);
  virtual ~GPFCluster() {;}
    
  double           getEnergy() {return en_;}
  virtual void     draw();
  void             setColor(int newcolor);
  void             setInitialColor();
    
  //overridden ROOT method 
  virtual void     Print();     // *MENU*
  virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    
  const GPFCluster& operator=( const GPFCluster& other ) {
    clus_ = other.clus_;
    return *this;
  }

 private:
  const reco::PFCluster*   clus_;
  //energy
  double                   en_;
    
};  
#endif
