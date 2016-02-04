#ifndef RecoTracker_TkSeedGenerator_SeedChargeSelector_h
#define RecoTracker_TkSeedGenerator_SeedChargeSelector_h
/* \class SeedChargeSelector
 *
 * \author Giuseppe Cerati, INFN
 *
 */

struct SeedChargeSelector {
  SeedChargeSelector( int charge ) : charge_( charge ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return ( t.startingState().parameters().charge() == charge_ ); 
  }

private:
  int charge_;
};

#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<SeedChargeSelector> {
      static SeedChargeSelector make( const edm::ParameterSet & cfg ) {
        return SeedChargeSelector(cfg.getParameter<int>( "charge" )); 
      }
    };
    
  }
}

#endif
