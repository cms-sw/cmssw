#ifndef CommonTools_Utils_DeltaPhiMinPairSelector_h
#define CommonTools_Utils_DeltaPhiMinPairSelector_h
/* \class DeltaPhiMinPairSelector
 *
 * \author Luca Lista, INFN
 *
 * $Id: DeltaPhiMinPairSelector.h,v 1.1 2007/06/19 15:39:08 llista Exp $
 */
#include "DataFormats/Math/interface/deltaPhi.h"

struct DeltaPhiMinPairSelector {
  DeltaPhiMinPairSelector( double deltaPhiMin ) : 
    deltaPhiMin_( deltaPhiMin ) { }
  template<typename T1, typename T2>
  bool operator()( const T1 & t1, const T2 & t2 ) const { 
    return deltaPhi( t1.phi(), t2.phi() ) > deltaPhiMin_;
  }

private:
  double deltaPhiMin_;
};


#endif
