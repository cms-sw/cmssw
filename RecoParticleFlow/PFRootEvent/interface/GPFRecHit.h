#ifndef Graphic_PFRecHit_h
#define Graphic_PFRecHit_h

/*! \file interface/GPFRecHit.h
    class to create graphic  object
    from physical object PFRecHit
*/  
 
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "TGraph.h"
#include <string>


class GPFRecHit : public TGraph {
  public:
    GPFRecHit() ;
    GPFRecHit(reco::PFRecHit *hit,int size,
              double *x,double *y , int color, std::string option);
    virtual ~GPFRecHit() {;}
    
    //override ROOT methods
    virtual void     Print();     // *MENU*
    virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual void     Draw();
    
  private:
    reco::PFRecHit *recHit_;
    // Draw option
    std::string     option_;
};  
#endif
