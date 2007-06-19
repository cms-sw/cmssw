#ifndef Utilities_DeltaRMinPairSelector_h
#define Utilities_DeltaRMinPairSelector_h
/* \class DeltaRMinPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: AnyPairSelector.h,v 1.3 2007/06/18 18:33:53 llista Exp $
 */
#include "DataFormats/Math/interface/deltaR.h"

struct DeltaRMinPairSelector {
  DeltaRMinPairSelector( double deltaRMin ) : 
    deltaRMin2_( deltaRMin * deltaRMin ) { }
  template<typename T1, typename T2>
  bool operator()( const T1 & t1, const T2 & t2 ) const { 
    return deltaR2( t1, t2 ) > deltaRMin2_;
  }

private:
  double deltaRMin2_;
};


#endif
