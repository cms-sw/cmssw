#ifndef CMSCGENnorm_h
#define CMSCGENnorm_h
//
// CMSCGENnorm.h       P. Biallass 2006  
//
// code based on l3cgen.f by T.Hebbeker
// implemented in CMSSW by P. Biallass 11.04.2006  
//
/////////////////////////////////////////////////////////////////////////////////
//
//  calculate normalisation factor (if statistics sufficient):
// 
//
//  flux dN/dOmega/dE/dA/dt at 100 GeV and zenith angle 0 (per sr-> theta<32.77 deg): 
//    2.63 * E-3 +- 0.06 * E-3 / m**2 / sr / GeV / s   (see Biallass+Hebbeker internal note 2007 "Improved Parametrization of the Cosmic Muon Flux for the generator CMSCGEN")
//
//  percentage of rejected events and surface needs to be corrected for later!
//
// for this we also need to:
//
// count muons with energy 100 GeV
//   and those with 1 - cos(theta) < 1/(2pi)  (-> gives directly dN/dOmega/dE !) (this means theta<32.77deg)
// 


#include "TMath.h" 
#include <iostream>
#include "GeneratorInterface/CosmicMuonGenerator/interface/CosmicMuonParameters.h"


class CMSCGENnorm 
{

private:

  int n100;
  int n100cos;
  int n;
  float flux;
  float Nnorm;

public:

  // constructor
  CMSCGENnorm(){
    n100=0; 
    n100cos=0;
}

  //destructor
  ~CMSCGENnorm(){
    n100=0; 
    n100cos=0;
}

  int events_n100cos(double energy, double theta); //count number of cosmic with energy 100 GeV and those with 1 - cos(theta) < 1/(2pi)
 
  float norm(int n100cos); //normalize to known flux
    
};
#endif

