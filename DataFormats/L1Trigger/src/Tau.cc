
#include "DataFormats/L1Trigger/interface/Tau.h"

l1t::Tau::Tau( const LorentzVector& p4,
	       int pt,
	       int eta,
	       int phi,
	       int qual,
	       int iso )
  : L1Candidate(p4, pt, eta, phi, qual,iso)
{
  
}

l1t::Tau::~Tau() 
{

}
