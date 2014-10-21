
#include "DataFormats/L1TCalorimeter/interface/CaloStage1Cluster.h"

l1t::CaloStage1Cluster::CaloStage1Cluster( const LorentzVector p4, 
			     int pt,
			     int eta,
			     int phi,
			     int qual)
  : L1Candidate(p4, pt, eta, phi, qual)
{
  
}

l1t::CaloStage1Cluster::~CaloStage1Cluster() 
{
  
}

