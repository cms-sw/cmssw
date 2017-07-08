#ifndef _COMMONDETUNIT_GEOMDETENUMERATORS_H_ 
#define _COMMONDETUNIT_GEOMDETENUMERATORS_H_

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include <iosfwd>

/** Global enumerators for Det types.
 */
namespace GeomDetEnumerators {
  enum Location {barrel, endcap, invalidLoc};
  enum SubDetector {PixelBarrel, PixelEndcap, TIB, TOB, TID, TEC, CSC, DT, RPCBarrel, RPCEndcap, GEM, ME0, P2OTB, P2OTEC, P1PXB, P1PXEC, P2PXB, P2PXEC, TimingBarrel, TimingEndcap, invalidDet};
  // gives subdetId in DetId conrrepsonding to the above
  constexpr unsigned int subDetId[21]={1,2,3,5,4,6, 0, 0,0,0,0,0,5,4,1,2,1,2, 2,2, 0}; // don't ask, don't ask, simply do not ask!
  //inverse (only for tracker)
  constexpr SubDetector tkDetEnum[8]={invalidDet, PixelBarrel, PixelEndcap, TIB, TID, TOB, TEC, invalidDet}; // don't ask, don't ask, simply do not ask!
  //to convert a physical subdetector (it could be larger than 5) into a geometry subdetector (between 0 and 5). ONLY for tracker
  constexpr SubDetector subDetGeom[21]={PixelBarrel, PixelEndcap, TIB, TOB, TID, TEC, invalidDet, invalidDet, invalidDet, invalidDet, invalidDet, invalidDet, TOB, TID, PixelBarrel, PixelEndcap, PixelBarrel, PixelEndcap, invalidDet, invalidDet, invalidDet};

bool isBarrel(GeomDetEnumerators::SubDetector m);
bool isEndcap(GeomDetEnumerators::SubDetector m);

bool isTrackerPixel(GeomDetEnumerators::SubDetector m);
bool isTrackerStrip(GeomDetEnumerators::SubDetector m);
bool isTracker(GeomDetEnumerators::SubDetector m);
bool isDT(GeomDetEnumerators::SubDetector m);
bool isCSC(GeomDetEnumerators::SubDetector m);
bool isRPC(GeomDetEnumerators::SubDetector m);
bool isGEM(GeomDetEnumerators::SubDetector m);
bool isME0(GeomDetEnumerators::SubDetector m);
bool isMuon(GeomDetEnumerators::SubDetector m);
bool isTiming(GeomDetEnumerators::SubDetector m);
 
}

/* overload << for correct output of the enumerators 
 *  (e.g. to get "barrel" instead of "0")
 */
std::ostream& operator<<( std::ostream& s, GeomDetEnumerators::Location l);
std::ostream& operator<<( std::ostream& s, GeomDetEnumerators::SubDetector m);


#endif
