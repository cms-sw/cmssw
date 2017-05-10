#include "DataFormats/L1Trigger/interface/EtSum.h"

l1t::EtSum::EtSum( const LorentzVector& p4,
		   EtSumType type,
		   int pt,
		   int eta,
		   int phi,
		   int qual)
  : L1Candidate(p4, pt, eta, phi, qual),
      type_(type)
{
  
}

l1t::EtSum::~EtSum() 
{

}

void l1t::EtSum::setType(EtSumType type)
{
  type_ = type;
}

l1t::EtSum::EtSumType l1t::EtSum::getType()
{
  return type_;
}
