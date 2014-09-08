#include "DataFormats/L1Trigger/interface/CaloSpare.h"

l1t::CaloSpare::CaloSpare( const LorentzVector& p4,
		   CaloSpareType type,
		   int pt,
		   int eta,
		   int phi,
		   int qual)
  : L1Candidate(p4, pt, eta, phi, qual, 0),
      type_(type)
{

}

l1t::CaloSpare::CaloSpare( const PolarLorentzVector& p4,
		   CaloSpareType type,
		   int pt,
		   int eta,
		   int phi,
		   int qual)
  : L1Candidate(p4, pt, eta, phi, qual, 0),
      type_(type)
{

}

l1t::CaloSpare::~CaloSpare()
{

}

void l1t::CaloSpare::setType(CaloSpareType type)
{
  type_ = type;
}

l1t::CaloSpare::CaloSpareType l1t::CaloSpare::getType() const
{
  return type_;
}
