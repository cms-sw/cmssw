#ifndef PHYSICSTOOLS_PSETPTMINSELECTOR_H
#define PHYSICSTOOLS_PSETPTMINSELECTOR_H
// $Id: PSetSelectors.h,v 1.5 2006/04/04 11:12:48 llista Exp $#
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/CandAlgos/interface/CandSelectorBase.h"
#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"

namespace cand {
  namespace modules {
  
    class PtMinCandSelector : public CandSelectorBase {
    public:
      explicit PtMinCandSelector( const edm::ParameterSet & cfg ) :
	CandSelectorBase( cfg,
			  boost::shared_ptr<CandSelector> (
                            new PtMinSelector( cfg.getParameter<double>( "ptMin" ) ) ) ) {
      }
    };
    
    class MassWindowCandSelector : public CandSelectorBase {
    public:
      explicit MassWindowCandSelector( const edm::ParameterSet & cfg ) :
	CandSelectorBase( cfg,
			  boost::shared_ptr<CandSelector> (
			    new MassWindowSelector( cfg.getParameter<double>( "massMin" ), 
						    cfg.getParameter<double>( "massMax" ) ) ) ) {
      }
    };
    
  }
}

#endif
