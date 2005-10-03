#ifndef PHYSICSTOOLS_PTMINSELECTOR_H
#define PHYSICSTOOLS_PTMINSELECTOR_H
// $Id: PtMinSelector.h,v 1.1 2005/07/29 07:05:57 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"

class PtMinSelector {
public:
  explicit PtMinSelector( const edm::ParameterSet & parms ) :
    ptMin( parms.getParameter<double>( "ptMin" ) ) {
  }
  bool operator()( const phystools::Candidate * c ) const {
    return c->pt() > ptMin;
  }
private:
  double ptMin;
};

#endif
