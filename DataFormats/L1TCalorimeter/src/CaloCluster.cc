
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

l1t::CaloCluster::CaloCluster( const LorentzVector p4, 
			     int pt,
			     int eta,
			     int phi,
			     int qual)
  : L1Candidate(p4, pt, eta, phi, qual)
{
  
}

l1t::CaloCluster::~CaloCluster() 
{
  
}

