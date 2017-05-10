
#include "DataFormats/L1Trigger/interface/Jet.h"

l1t::Jet::Jet( const LorentzVector& p4,
	       int pt,
	       int eta,
	       int phi,
	       int qual )
  : L1Candidate(p4, pt, eta, phi, qual)
{
  
}

l1t::Jet::~Jet() 
{

}
