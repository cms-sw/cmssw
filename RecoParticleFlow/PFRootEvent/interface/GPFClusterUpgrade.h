#ifndef Graphic_PFClusterUpgrade_h
#define Graphic_PFClusterUpgrade_h

/*! \file interface/GPFCluster.h
  class to create graphic object
  from physical object PFCluster
*/  
 
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBaseUpgrade.h" 
#include "TMarker.h"


class GPFClusterUpgrade : public GPFBaseUpgrade, public TMarker {
 public:

  GPFClusterUpgrade(DisplayManagerUpgrade *dm, int view, int ident, 
             const reco::PFCluster* clus,
             double x,double y,TAttMarker *attm);
  virtual ~GPFClusterUpgrade() {;}
    
  double           getEnergy() {return en_;}
  virtual void     draw();
  void             setColor();
  void             setColor(int newcol);
  void             setInitialColor();
  void             setNewStyle();
  void             setNewSize(); 
    
  //overridden ROOT method 
  virtual void     Print();     // *MENU*
  virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    
  const GPFClusterUpgrade& operator=( const GPFClusterUpgrade& other ) {
    clus_ = other.clus_;
    return *this;
  }

 private:
  const reco::PFCluster*   clus_;
  //energy
  double                   en_;
    
};  
#endif
