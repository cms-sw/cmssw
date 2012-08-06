#ifndef Graphic_PFCandidateUpgrade_h
#define Graphic_PFCandidateUpgrade_h

/*! \file interface/GPFCandidate.h
  class to create graphic  object
  from physical object PFCandidate
*/  
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "RecoParticleFlow/PFRootEvent/interface/GPFBaseUpgrade.h" 
#include "TGraph.h"
#include <string>


class GPFCandidateUpgrade : public GPFBaseUpgrade, public TGraph {
 public:
  GPFCandidateUpgrade(DisplayManagerUpgrade *dm,int view, int ident, 
            reco::PFCandidate *candidate,int size,
            double *x,double *y , int color, std::string option);
  virtual ~GPFCandidateUpgrade() {}
    
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
  reco::PFCandidate *candidate_;

  /// energy
  double          en_;

  /// root draw option
  std::string     option_;
    
};  
#endif
