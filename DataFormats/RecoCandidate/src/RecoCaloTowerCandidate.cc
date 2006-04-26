// $Id: RecoCaloTowerCandidate.cc,v 1.1 2006/02/28 10:59:15 llista Exp $
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
  const RecoCandidate * dstc = dynamic_cast<const RecoCandidate *>( & c );
  if ( dstc == 0 ) return false;
  CaloTowerRef c1 = caloTower(), c2 = dstc->caloTower();
  if ( ! c1.isNull() && ! c2.isNull() && c1 == c2 ) return true;
  return false;
}
