#ifndef DataFormats_PortableTestObjects_interface_HitHostCollection_h
#define DataFormats_PortableTestObjects_interface_HitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/HitSoA.h"

namespace portabletest {

  using HitHostCollection = PortableHostCollection<HitSoA>;
  using HitToTrackHostCollection = PortableHostCollection<HitToTrackSoA>;
  using TrackBeginHostCollection = PortableHostCollection<TrackBeginSoA>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_HitHostCollection_h
