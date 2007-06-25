#ifndef Graphic_PFPart_h
#define Graphic_PFPart_h

/*! \file interface/GPFPart.h
    class to create graphic object
    from physical object of class PFSimpleParticle
*/  
 
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"

#include "TGraph.h"
#include <string>


class GPFPart : public TGraph {
  public:
    GPFPart() ;
    GPFPart(const reco::PFSimParticle *ptc, int size, double *x, double *y,
            int linestyle, int markerstyle, double markersize, int color,
	    std::string option);
	     
    virtual ~GPFPart() {;}
    
    //override ROOT method
    virtual void     Print();     // *MENU*
    virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual void     Draw();
    
    const GPFPart& operator=( const GPFPart& other ) {
      part_ = other.part_;
      return *this;
    }
    
  private:
    const reco::PFSimParticle*  part_;
    //draw option
    std::string                 option_;
    
      
};  
#endif
