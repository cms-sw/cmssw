#ifndef RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageHost_h
#define RecoLocalTracker_SiPixelClusterizer_interface_SiPixelImageHost_h

#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

using SiPixelImageHost = PortableHostCollection<SiPixelImageSoA>;

#endif
