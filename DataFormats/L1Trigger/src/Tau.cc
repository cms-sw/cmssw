
#include "DataFormats/L1Trigger/interface/Tau.h"

l1t::Tau::Tau( const LorentzVector& p4,
	       int pt,
	       int eta,
	       int phi,
	       int qual,
	       int iso )
  : L1Candidate(p4, pt, eta, phi, qual),
    hwIso_(iso)
{
  
}

l1t::Tau::~Tau() 
{

}

void l1t::Tau::setHwIso(int iso)
{
  hwIso_ = iso;
}

int l1t::Tau::hwIso()
{
  return hwIso_;
}
