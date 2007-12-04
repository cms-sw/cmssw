#ifndef Graphic_PFSimPart_h
#define Graphic_PFSimPart_h

/*! \file interface/GPFSimParticle.h
    class to create graphic object
    from physical object of class PFSimpleParticle
*/  
 
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h" 

#include "TGraph.h"
#include <string>


class GPFSimParticle : public GPFBase, public TGraph {
  public:
    GPFSimParticle() ;
    GPFSimParticle(DisplayManager *dm,int view, int ident, const reco::PFSimParticle *ptc, int size, double *x, double *y,
            double pt,int markerstyle,std::string option);
	     
    virtual ~GPFSimParticle() {;}
    
    virtual void     draw();
    double           getPt() { return pt_;}
    void             setColor(int newcolor);
    void             setInitialColor();
     
    //overridden ROOT method
    virtual void     Print();     // *MENU*
    virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    //
    const GPFSimParticle& operator=( const GPFSimParticle& other ) {
      part_ = other.part_;
      return *this;
    }
    
  private:
    const reco::PFSimParticle*  part_;
    double                      pt_;
    //draw option
    std::string                 option_;
    //initial color 
    int                         color_;
      
};  
#endif
