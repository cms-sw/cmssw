#ifndef _COMMONDETUNIT_GEOMDETENUMERATORS_H_ 
#define _COMMONDETUNIT_GEOMDETENUMERATORS_H_

#include <iosfwd>
#include <ostream>

/** Global enumerators for Det types.
 */
namespace GeomDetEnumerators {
  enum Location {barrel, endcap};
  enum SubDetector {PixelBarrel, PixelEndcap, TIB, TOB, TID, TEC, CSC, DT, RPCBarrel, RPCEndcap};
}

/* overload << for correct output of the enumerators 
 *  (e.g. to get "barrel" instead of "0")
 */
std::ostream& operator<<( std::ostream& s, GeomDetEnumerators::Location l);
std::ostream& operator<<( std::ostream& s, GeomDetEnumerators::SubDetector m);


#endif
