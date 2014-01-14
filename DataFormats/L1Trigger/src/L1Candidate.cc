
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

l1t::L1Candidate::L1Candidate(){}

l1t::L1Candidate::L1Candidate( const LorentzVector& p4,
			       int pt,
			       int eta,
			       int phi,
			       int qual )
  : LeafCandidate( ( char ) 0, p4 ),
    hwPt_(pt),
    hwEta_(eta),
    hwPhi_(phi),
    hwQual_(qual)
{

}

l1t::L1Candidate::L1Candidate( const PolarLorentzVector& p4,
			       int pt,
			       int eta,
			       int phi,
			       int qual )
  : LeafCandidate( ( char ) 0, p4 ),
    hwPt_(pt),
    hwEta_(eta),
    hwPhi_(phi),
    hwQual_(qual)
{

}

l1t::L1Candidate::~L1Candidate()
{

}

void l1t::L1Candidate::setHwPt(int pt)
{
  hwPt_ = pt;
}

void l1t::L1Candidate::setHwEta(int eta)
{
  hwEta_ = eta;
}

void l1t::L1Candidate::setHwPhi(int phi)
{
  hwPhi_ = phi;
}

void l1t::L1Candidate::setHwQual(int qual)
{
  hwQual_ = qual;
}

int l1t::L1Candidate::hwPt() const
{
  return hwPt_;
}

int l1t::L1Candidate::hwEta() const
{
  return hwEta_;
}

int l1t::L1Candidate::hwPhi() const
{
  return hwPhi_;
}


int l1t::L1Candidate::hwQual() const
{
  return hwQual_;
}
