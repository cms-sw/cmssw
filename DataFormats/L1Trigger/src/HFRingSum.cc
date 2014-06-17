#include "DataFormats/L1Trigger/interface/HFRingSum.h"

l1t::HFRingSum::HFRingSum( const LorentzVector& p4,
		   HFRingSumType type,
		   int pt,
		   int eta,
		   int phi,
		   int qual)
  : L1Candidate(p4, pt, eta, phi, qual, 0),
      type_(type)
{

}

l1t::HFRingSum::HFRingSum( const PolarLorentzVector& p4,
		   HFRingSumType type,
		   int pt,
		   int eta,
		   int phi,
		   int qual)
  : L1Candidate(p4, pt, eta, phi, qual, 0),
      type_(type)
{

}

l1t::HFRingSum::~HFRingSum()
{

}

void l1t::HFRingSum::setType(HFRingSumType type)
{
  type_ = type;
}

l1t::HFRingSum::HFRingSumType l1t::HFRingSum::getType() const
{
  return type_;
}
