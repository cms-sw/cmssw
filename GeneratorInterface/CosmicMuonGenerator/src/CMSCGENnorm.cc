//
// CMSCGENnorm.cc       P. Biallass 2006
//
// code based on l3cgen.f by T.Hebbeker
// implemented in CMSSW by P. Biallass 11.04.2006
// see header for documentation
//

#include "GeneratorInterface/CosmicMuonGenerator/interface/CMSCGENnorm.h"

//count muons which are around 100GeV and theta<33 deg (~vertical), do this for EVERY generated cosmic

int CMSCGENnorm::events_n100cos(double energy, double theta) {
  if (energy > 99.5 && energy < 100.5) {
    n100 = n100 + 1;

    if (1. - cos(theta) < 1. / (2. * Pi)) {  //theta is in rad
      n100cos = n100cos + 1;
    }
  }
  return n100cos;
}

// determine normalization using known flux, do this in the end. Percentage of rejected events and surface needs to be corrected for later!
// Note that the number of actually DICED cosmics is needed for normalisation, and sufficient statistics to have muons at 100 GeV!

float CMSCGENnorm::norm(int n100cos) {
  flux = 2.63e-3;  // +- 0.06e-3 [1/m**2/sr/GeV/s]

  n = n100cos;  // [1/sr/GeV]

  //rate=N/runtime --> Nnorm ~ (1/runtime/m^2 at surface plane) as rate corresponds to known flux
  Nnorm = flux / n;

  //err of Nnorm = Nnorm* 1/sqrt(n)

  return Nnorm;
}
