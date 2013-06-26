#ifndef Graphic_PFSimPart_h
#define Graphic_PFSimPart_h

/*! \file interface/GPFSimParticle.h
  class to create graphic object
  from physical object of class PFSimpleParticle
*/  
 
#include "DataFormats/ParticleFlowReco/interface/PFSimParticle.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h" 

#include "TGraph.h"
#include <string>


class GPFSimParticle : public GPFBase, public TGraph {
 public:

  GPFSimParticle(DisplayManager *dm,int view, int ident, 
                 const reco::PFSimParticle *ptc, 
                 int size, double *x, double *y,
                 double pt,TAttMarker *attm,TAttLine *attl,
                 std::string option);
                 
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

    
 private:
  const reco::PFSimParticle*  part_;
  double                      pt_;
  //draw option
  std::string                 option_;
      
};  
#endif
