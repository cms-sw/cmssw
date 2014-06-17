#include "DataFormats/L1Trigger/interface/HFBitCount.h"

l1t::HFBitCount::HFBitCount( const LorentzVector& p4,
		   HFBitCountType type,
		   int pt,
		   int eta,
		   int phi,
		   int qual)
  : L1Candidate(p4, pt, eta, phi, qual, 0),
      type_(type)
{

}

l1t::HFBitCount::HFBitCount( const PolarLorentzVector& p4,
		   HFBitCountType type,
		   int pt,
		   int eta,
		   int phi,
		   int qual)
  : L1Candidate(p4, pt, eta, phi, qual, 0),
      type_(type)
{

}

l1t::HFBitCount::~HFBitCount()
{

}

void l1t::HFBitCount::setType(HFBitCountType type)
{
  type_ = type;
}

l1t::HFBitCount::HFBitCountType l1t::HFBitCount::getType() const
{
  return type_;
}
