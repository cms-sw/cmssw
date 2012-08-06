#ifndef Graphic_PFTrackUpgrade_h
#define Graphic_PFTrackUpgrade_h

/*! \file interface/GPFRecTrack.h
  class to create graphic  object
  from physical object of class PFRecTrack
*/  
 
#include "RecoParticleFlow/PFRootEvent/interface/GPFBaseUpgrade.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "TGraph.h"
#include <string>


class GPFTrackUpgrade : public GPFBaseUpgrade, public TGraph {
 public:
  GPFTrackUpgrade(DisplayManagerUpgrade *dm,int view, int ident, 
           reco::PFRecTrack *tra, int size, double *x, double *y,
           double pt,TAttMarker *attm,TAttLine *attl, std::string option);      

  virtual ~GPFTrackUpgrade() {;}
    
  virtual void     draw();
  double           getPt() { return pt_;}
  void             setColor();
  void             setColor(int newcol);
  void             setInitialColor();
  void             setNewStyle();
  void             setNewSize();
    
  //overridden ROOT method
  virtual void     Print();     // *MENU*
  virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
  //
    
  const GPFTrackUpgrade& operator=( const GPFTrackUpgrade& other ) {
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
