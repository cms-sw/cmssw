#ifndef IsolationAlgos_PtIsolationAlgo_h
#define IsolationAlgos_PtIsolationAlgo_h
/* Partial spacialization of parameter set adapeter helper
 *
 */
#include "PhysicsTools/IsolationUtils/interface/PtIsolationAlgo.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  namespace modules {
    
    template<typename T, typename C> 
    struct ParameterAdapter<PtIsolationAlgo<T, C> > { 
      static PtIsolationAlgo<T, C> make( const edm::ParameterSet & cfg ) {
	  return PtIsolationAlgo<T, C>( cfg.template getParameter<double>( "dRMin" ), 
					cfg.template getParameter<double>( "dRMax" ),
					cfg.template getParameter<double>( "dzMax" ),
					cfg.template getParameter<double>( "d0Max" ),
					cfg.template getParameter<double>( "ptMin" ) );
      }
    };
  }
}

#endif
