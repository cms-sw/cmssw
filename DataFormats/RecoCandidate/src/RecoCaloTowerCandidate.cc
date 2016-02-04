// $Id: RecoCaloTowerCandidate.cc,v 1.5 2006/05/31 12:45:46 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"

using namespace reco;

RecoCaloTowerCandidate::~RecoCaloTowerCandidate() { }

RecoCaloTowerCandidate * RecoCaloTowerCandidate::clone() const { 
  return new RecoCaloTowerCandidate( * this ); 
}

CaloTowerRef RecoCaloTowerCandidate::caloTower() const {
  return caloTower_;
}

bool RecoCaloTowerCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 &&  
	   checkOverlap( caloTower(), o->caloTower() ) 
	   );
}
