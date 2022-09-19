#ifndef DataFormats_TrackReco_TrackPositionStorage_h
#define DataFormats_TrackReco_TrackPositionStorage_h
// -*- C++ -*-
//
// Package:     DataFormats/TrackReco
// Class  :     TrackPositionStorage
//
/**\class TrackPositionStorage TrackPositionStorage.h "TrackPositionStorage.h"

 Description: Floating point lossy compressed cartesian 3d

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 12 Sep 2022 15:12:10 GMT
//

// system include files

// user include files
#include "Rtypes.h"

// forward declarations
namespace reco {
  namespace storage {
    struct TrackPositionValues {
      TrackPositionValues() : fX(0.), fY(0.), fZ(0.) {}
      Double32_t fX;  //[-1100,1100,24]
      Double32_t fY;  //[-1100,1100,24]
      Double32_t fZ;
    };

    struct TrackPositionStorage {
    public:
      TrackPositionValues fCoordinates;
    };
  }  // namespace storage
}  // namespace reco
#endif
