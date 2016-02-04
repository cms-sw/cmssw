#ifndef IsolationAlgos_CalIsolationAlgoNoExtrapol_h
#define IsolationAlgos_CalIsolationAlgoNoExtrapol_h
/* Partial spacialization of parameter set adapeter helper
 *
 */
#include "PhysicsTools/IsolationUtils/interface/CalIsolationAlgoNoExp.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

namespace reco {
  namespace modules {
    
    template<typename T, typename C> 
    struct ParameterAdapter<CalIsolationAlgo<T, C> > { 
      static CalIsolationAlgo<T, C> make( const edm::ParameterSet & cfg ) {
	  return CalIsolationAlgoNoExp<T, C>( cfg.template getParameter<double>( "dRMin" ), 
					      cfg.template getParameter<double>( "dRMax" ) );
      }
    };
  }
}

#endif
