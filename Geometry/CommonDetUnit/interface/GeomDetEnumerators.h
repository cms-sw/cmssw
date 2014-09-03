#ifndef _COMMONDETUNIT_GEOMDETENUMERATORS_H_ 
#define _COMMONDETUNIT_GEOMDETENUMERATORS_H_

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include <iosfwd>

/** Global enumerators for Det types.
 */
namespace GeomDetEnumerators {
  enum Location {barrel, endcap, invalidLoc};
  enum SubDetector {PixelBarrel, PixelEndcap, TIB, TOB, TID, TEC, CSC, DT, RPCBarrel, RPCEndcap, GEM, ME0, invalidDet};
  // gives subdetId in DetId conrrepsonding to the above
  constexpr unsigned int subDetId[13]={1,2,3,5,4,6, 0, 0,0,0,0,0, 0}; // don't ask, don't ask, simply do not ask!
  //inverse (only for tracker)
  constexpr SubDetector tkDetEnum[8]={invalidDet, PixelBarrel, PixelEndcap, TIB, TID, TOB, TEC, invalidDet}; // don't ask, don't ask, simply do not ask!
 
}

/* overload << for correct output of the enumerators 
 *  (e.g. to get "barrel" instead of "0")
 */
std::ostream& operator<<( std::ostream& s, GeomDetEnumerators::Location l);
std::ostream& operator<<( std::ostream& s, GeomDetEnumerators::SubDetector m);


#endif
