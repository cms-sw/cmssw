#ifndef Graphic_GenParticle_h
#define Graphic_GenParticle_h

/*! \file interface/GPFGenParticle.h
  class to create graphic object
  to represent GenParticle
*/ 

#include <string> 
#include "RecoParticleFlow/PFRootEvent/interface/GPFBase.h" 
#include "TMarker.h"
#include "TLatex.h"
#include "TLine.h"

class GPFGenParticle : public GPFBase, public TMarker, public TLatex {
 public:
  GPFGenParticle(DisplayManager *dm, int view, int ident,
                 double eta, double phi, double en, double pt,int barcode,
                 TAttMarker *attm, std::string name,std::string latexName);
  GPFGenParticle(DisplayManager *dm, int view, int ident,
                 double *eta, double *phi, double en, double pt,
                 int barcode, int barcodeMother,
                 TAttMarker *attm, std::string name,std::string latexName);
                   
                   
                   
  virtual ~GPFGenParticle() {;}
    
  double   getEnergy() { return en_;}
  double   getPt()     { return pt_;}
  virtual void     draw();
  void             setColor();
  void             setColor(int newcol);
  void             setInitialColor();
  void             setNewStyle();
  void             setNewSize(); 
    
  //overridden ROOT method 
  virtual void     Print();     // *MENU*
  virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
    
    
 private:
  //energy
  double                   en_;
  double                   pt_;
  std::string              name_;
  int                      barcode_;
  int                      barcodeMother_;
  TLine *                  line_;
  
    
};  
#endif
                    
