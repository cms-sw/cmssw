#ifndef DataFormats_PixelCluster_classes_h
#define DataFormats_PixelCluster_classes_h
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    SiPixelClusterCollection c1;
    edm::Wrapper<SiPixelClusterCollection> w1;
    SiPixelClusterRef r1;
    // warning: RefVector dictionary does not compile
    //    SiPixelClusterRefVector rv1;
    SiPixelClusterRefProd rp1;
  }
}

#endif // SIPIXELCLUSTER_CLASSES_H
