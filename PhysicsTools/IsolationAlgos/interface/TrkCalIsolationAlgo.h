#ifndef IsolationAlgos_TrkCalIsolationAlgo_h
#define IsolationAlgos_TrkCalIsolationAlgo_h
/* Partial spacialization of parameter set adapeter helper
 *
 */
#include "PhysicsTools/IsolationUtils/interface/TrkCalIsolationAlgo.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  namespace modules {
    
    template<typename T, typename C> 
    struct ParameterAdapter<TrkCalIsolationAlgo<T, C> > { 
      static TrkCalIsolationAlgo<T, C> make( const edm::ParameterSet & cfg ) {
	  return TrkCalIsolationAlgo<T, C>( cfg.template getParameter<double>( "dRMin" ), 
					    cfg.template getParameter<double>( "dRMax" ) );
      }
    };
  }
}

#endif
