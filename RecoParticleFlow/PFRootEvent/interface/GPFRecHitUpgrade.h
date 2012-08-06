#ifndef Graphic_PFRecHitUpgrade_h
#define Graphic_PFRecHitUpgrade_h

/*! \file interface/GPFRecHit.h
  class to create graphic  object
  from physical object PFRecHit
*/  
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBaseUpgrade.h" 
#include "TGraph.h"
#include <string>


class GPFRecHitUpgrade : public GPFBaseUpgrade, public TGraph {
 public:
  GPFRecHitUpgrade(DisplayManagerUpgrade *dm,int view, int ident, 
            reco::PFRecHit *hit,int size,
            double *x,double *y , int color, std::string option);
  virtual ~GPFRecHitUpgrade() {}
    
  virtual void     draw();
  double           getEnergy() { return en_;}
  std::string      getOption() { return option_;}
  void             setColor();
  void             setColor(int newcolor);
  void             setInitialColor();
  void             setNewStyle();
  void             setNewSize();
    
  //overridden ROOT methods
  virtual void     Print();     // *MENU*
  virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    
 private:
  reco::PFRecHit *recHit_;

  /// energy
  double          en_;

  /// root draw option
  std::string     option_;
    
};  
#endif
