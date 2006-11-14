// $Id: CaloRecHitCandidate.cc,v 1.5 2006/05/31 12:45:46 llista Exp $
#include "DataFormats/RecoCandidate/interface/CaloRecHitCandidate.h"

using namespace reco;

CaloRecHitCandidate::~CaloRecHitCandidate() { }

CaloRecHitCandidate * CaloRecHitCandidate::clone() const { 
  return new CaloRecHitCandidate( * this ); 
}

bool CaloRecHitCandidate::overlap( const Candidate & c ) const {
  const CaloRecHitCandidate * o = dynamic_cast<const CaloRecHitCandidate *>( & c );
  if ( o == 0 ) return false;
  if ( caloRecHit().isNull() ) return false;
  if ( o->caloRecHit().isNull() ) return false;
  return ( caloRecHit() != o->caloRecHit() );
}
