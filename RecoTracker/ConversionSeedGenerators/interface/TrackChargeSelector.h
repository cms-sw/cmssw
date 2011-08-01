#ifndef TrackingForConversion_TrackChargeSelector_h
#define TrackingForConversion_TrackChargeSelector_h
/* \class TrackChargeSelector
 *
 * \author Domenico Giordano, CERN
 *
 */

struct TrackChargeSelector {
  TrackChargeSelector( int charge ) : charge_( charge ) { }
  template<typename T>
  bool operator()( const T & t ) const { 
    return ( t.charge() == charge_ ); 
  }

private:
  int charge_;
};

#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<TrackChargeSelector> {
      static TrackChargeSelector make( const edm::ParameterSet & cfg ) {
        return TrackChargeSelector(cfg.getParameter<int>( "charge" )); 
      }
    };
    
  }
}

#endif
