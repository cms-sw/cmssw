#ifndef Graphic_PFTrack_h
#define Graphic_PFTrack_h

/*! \file interface/GPFRecTrack.h
    class to create graphic recHit object
    from physical object of class PFRecTrack
*/  
 
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "TGraph.h"
#include <string>


class GPFTrack : public TGraph {
  public:
    GPFTrack() ;
    GPFTrack(reco::PFRecTrack *tra, int size, double *x, double *y,
             double pt,int linestyle,std::string option);	     
    virtual ~GPFTrack() {;}
    
    //override ROOT method
    virtual void     Print();     // *MENU*
    virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual void     Draw();
    double           getPt() { return pt_;}
    
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
