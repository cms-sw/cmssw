//
// $Id:$
//

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"

using namespace reco;

Centrality::Centrality(double energy, int bin, float npart, float npart_sig , float ncoll, float ncoll_sig, float b, float b_sig)
  : 
HFEnergy_(energy),
Bin_(bin), 
Npart_(npart), 
Npart_sigma_(npart_sig), 
Ncoll_(ncoll), 
Ncoll_sigma_(ncoll_sig), 
imp_par_(b), 
imp_par_sigma_(b_sig)
{
  // default constructor
}


Centrality::~Centrality()
{
}


