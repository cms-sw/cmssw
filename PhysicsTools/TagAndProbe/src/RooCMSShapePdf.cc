
#include "RooAbsPdf.h"
#include "RooMath.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include "TFile.h"
#include "PhysicsTools/TagAndProbe/interface/RooCMSShapePdf.h"

 RooCMSShapePdf::RooCMSShapePdf(const char *name, const char *title, 
                        RooAbsReal& _x,
                        RooAbsReal& _alpha,
                        RooAbsReal& _beta,
                        RooAbsReal& _gamma,
                        RooAbsReal& _peak) :
   RooAbsPdf(name,title), 
   x("x","x",this,_x),
   alpha("alpha","alpha",this,_alpha),
   beta("beta","beta",this,_beta),
   gamma("gamma","gamma",this,_gamma),
   peak("peak","peak",this,_peak)
 { 
 } 


 RooCMSShapePdf::RooCMSShapePdf(const RooCMSShapePdf& other, const char* name) :  
   RooAbsPdf(other,name), 
   x("x",this,other.x),
   alpha("alpha",this,other.alpha),
   beta("beta",this,other.beta),
   gamma("gamma",this,other.gamma),
   peak("peak",this,other.peak)
 { 
 } 



 Double_t RooCMSShapePdf::evaluate() const 
 { 
  // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE 

  //Double_t erf = TMath::Erfc((alpha - x) * beta);
  Double_t erf = RooMath::erfc((alpha - x) * beta);
  Double_t u = (x - peak)*gamma;

  if(u < -70) u = 1e20;
  else if( u>70 ) u = 0;
  else u = exp(-u);   //exponential decay
  return erf*u;
 } 

