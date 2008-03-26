#ifndef Graphic_PFTrack_h
#define Graphic_PFTrack_h

/*! \file interface/GPFRecTrack.h
  class to create graphic  object
  from physical object of class PFRecTrack
*/  
 
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "TGraph.h"
#include <string>


class GPFTrack : public GPFBase, public TGraph {
 public:
  GPFTrack(DisplayManager *dm,int view, int ident, 
	   reco::PFRecTrack *tra, int size, double *x, double *y,
           double pt,int linestyle, int linecolor, std::string option);      

  virtual ~GPFTrack() {;}
    
  virtual void     draw();
  double           getPt() { return pt_;}
  void             setColor(int newcolor );
  void             setInitialColor();
    
  //overridden ROOT method
  virtual void     Print();     // *MENU*
  virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
  //
    
  const GPFTrack& operator=( const GPFTrack& other ) {
    track_ = other.track_;
    return *this;
  }
    
 private:
  //draw option
  reco::PFRecTrack*  track_;
  double             pt_;
  std::string        option_;

    
      
};  
#endif
