#ifndef PHYSICSTOOLS_PTMINSELECTOR_H
#define PHYSICSTOOLS_PTMINSELECTOR_H
// $Id: PtMinSelector.h,v 1.3 2005/07/15 08:41:28 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"

namespace phystools {
  
  class Candidate;
  
  class PtMinSelector {
  public:
    explicit PtMinSelector( const edm::ParameterSet & parms ) :
      ptMin( parms.getParameter<double>( "ptMin" ) ) {
    }
    bool operator()( const Candidate * c ) const {
      return c->pt() > ptMin;
    }
  private:
    double ptMin;
  };
  
}

#endif
