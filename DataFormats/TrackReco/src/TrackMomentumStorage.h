#ifndef DataFormats_TrackReco_TrackMomentumStorage_h
#define DataFormats_TrackReco_TrackMomentumStorage_h
// -*- C++ -*-
//
// Package:     DataFormats/TrackReco
// Class  :     TrackMomentumStorage
//
/**\class TrackMomentumStorage TrackMomentumStorage.h "TrackMomentumStorage.h"

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
    struct TrackMomentumValues {
      TrackMomentumValues() : fX(0.), fY(0.), fZ(0.) {}
      Double32_t fX;  //[0,0,13]
      Double32_t fY;  //[0,0,13]
      Double32_t fZ;  //[0,0,13]
    };

    struct TrackMomentumStorage {
    public:
      TrackMomentumValues fCoordinates;
    };
  }  // namespace storage
}  // namespace reco
#endif
