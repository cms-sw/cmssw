#ifndef Graphic_PFSimPart_h
#define Graphic_PFSimPart_h

/*! \file interface/GPFSimParticle.h
    class to create graphic object
    from physical object of class PFSimpleParticle
*/  
 
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"

#include "TGraph.h"
#include <string>


class GPFSimParticle : public TGraph {
  public:
    GPFSimParticle() ;
    GPFSimParticle(const reco::PFSimParticle *ptc, int size, double *x, double *y,
            double pt,int markerstyle,std::string option);
	     
    virtual ~GPFSimParticle() {;}
    
    //override ROOT method
    virtual void     Print();     // *MENU*
    virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual void     Draw();
    double           getPt() { return pt_;}
    
    const GPFSimParticle& operator=( const GPFSimParticle& other ) {
      part_ = other.part_;
      return *this;
    }
    
  private:
    const reco::PFSimParticle*  part_;
    double                      pt_;
    //draw option
    std::string                 option_;
    
      
};  
#endif
