
#include "DataFormats/L1Trigger/interface/EGamma.h"

l1t::EGamma::EGamma( const LorentzVector& p4,
		     int pt,
		     int eta,
		     int phi,
		     int qual,
		     int iso )
  : L1Candidate(p4, pt, eta, phi, qual),
    hwIso_(iso)
{
  
}

l1t::EGamma::~EGamma() 
{

}

void l1t::EGamma::setHwIso(int iso)
{
  hwIso_ = iso;
}

int l1t::EGamma::hwIso()
{
  return hwIso_;
}
