
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

l1t::CaloEmCand::CaloEmCand( const LorentzVector& p4, 
			     int pt,
			     int eta,
			     int phi,
			     int qual)
  : L1Candidate(p4, pt, eta, phi, qual)
{
  
}

l1t::CaloEmCand::~CaloEmCand() 
{
  
}

