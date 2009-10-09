#ifndef Graphic_PFRecHit_h
#define Graphic_PFRecHit_h

/*! \file interface/GPFRecHit.h
  class to create graphic  object
  from physical object PFRecHit
*/  
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h" 
#include "TGraph.h"
#include <string>


class GPFRecHit : public GPFBase, public TGraph {
 public:
  GPFRecHit(DisplayManager *dm,int view, int ident, 
            reco::PFRecHit *hit,int size,
            double *x,double *y , int color, std::string option);
  virtual ~GPFRecHit() {}
    
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
