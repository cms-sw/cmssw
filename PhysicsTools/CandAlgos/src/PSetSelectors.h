#ifndef PHYSICSTOOLS_PSETPTMINSELECTOR_H
#define PHYSICSTOOLS_PSETPTMINSELECTOR_H
// $Id: PtMinSelector.h,v 1.3 2005/10/03 10:12:11 llista Exp $#
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"

class PSetPtMinSelector : public PtMinSelector {
public:
  explicit PSetPtMinSelector( const edm::ParameterSet & parms ) :
    PtMinSelector( parms.getParameter<double>( "ptMin" ) ) {
  }
};


class PSetMassWindowSelector : public MassWindowSelector {
public:
  explicit PSetMassWindowSelector( const edm::ParameterSet & parms ) :
    MassWindowSelector( parms.getParameter<double>( "massMin" ), 
			parms.getParameter<double>( "massMax" ) ) {
  }
};

#endif
