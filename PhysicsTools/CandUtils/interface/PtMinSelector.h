#ifndef PHYSICSTOOLS_PTMINSELECTOR_H
#define PHYSICSTOOLS_PTMINSELECTOR_H
// $Id: PtMinSelector.h,v 1.7 2005/10/25 09:08:31 llista Exp $
#include "PhysicsTools/CandUtils/interface/CandSelector.h"

class PtMinSelector : public CandSelector {
public:
  explicit PtMinSelector( double cut ) :
    ptMin( cut ) {
  }
  bool operator()( const reco::Candidate & c ) const;
private:
  double ptMin;
};

#endif
