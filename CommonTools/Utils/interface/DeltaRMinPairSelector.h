#ifndef CommonTools_Utils_DeltaRMinPairSelector_h
#define CommonTools_Utils_DeltaRMinPairSelector_h
/* \class DeltaRMinPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: DeltaRMinPairSelector.h,v 1.1 2009/02/24 14:40:26 llista Exp $
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
