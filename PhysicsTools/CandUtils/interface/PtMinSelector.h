#ifndef PHYSICSTOOLS_PTMINSELECTOR_H
#define PHYSICSTOOLS_PTMINSELECTOR_H
// $Id: PtMinSelector.h,v 1.2 2005/10/03 09:17:45 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"

class PtMinSelector {
public:
  explicit PtMinSelector( const edm::ParameterSet & parms ) :
    ptMin( parms.getParameter<double>( "ptMin" ) ) {
  }
  bool operator()( const aod::Candidate * c ) const {
    return c->pt() > ptMin;
  }
private:
  double ptMin;
};

#endif
