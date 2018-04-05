
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

l1t::L1Candidate::L1Candidate():hwPt_(0),hwEta_(0),hwPhi_(0),hwQual_(0),hwIso_(0){}


l1t::L1Candidate::L1Candidate( const LorentzVector& p4,
			       int pt,
			       int eta,
			       int phi,
			       int qual,
			       int iso )
  : LeafCandidate( ( char ) 0, p4 ),
    hwPt_(pt),
    hwEta_(eta),
    hwPhi_(phi),
    hwQual_(qual),
    hwIso_(iso)    
{

}

l1t::L1Candidate::L1Candidate( const PolarLorentzVector& p4,
			       int pt,
			       int eta,
			       int phi,
			       int qual,
			       int iso )
  : LeafCandidate( ( char ) 0, p4 ),
    hwPt_(pt),
    hwEta_(eta),
    hwPhi_(phi),
    hwQual_(qual),
    hwIso_(iso)
{

}

l1t::L1Candidate::~L1Candidate()
{

}

bool l1t::L1Candidate::operator==(const l1t::L1Candidate& rhs) const
{
  return hwPt_ == rhs.hwPt()
      && hwEta_ == rhs.hwEta()
      && hwPhi_ == rhs.hwPhi()
      && hwQual_ == rhs.hwQual()
      && hwIso_ == rhs.hwIso();
}

