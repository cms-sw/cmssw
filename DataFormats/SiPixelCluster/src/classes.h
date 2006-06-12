#ifndef DataFormats_PixelCluster_classes_h
#define DataFormats_PixelCluster_classes_h
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<SiPixelCluster> v1;
    edm::DetSet<SiPixelCluster> ds1;
    std::vector<edm::DetSet<SiPixelCluster> > vds1;
    SiPixelClusterCollection c1;
    edm::Wrapper<SiPixelClusterCollection> w1;
    SiPixelClusterRef r1;
    // warning: dictionary for SiPixelClusterRefVector does not work
    //    SiPixelClusterRefVector rv1;
    SiPixelClusterRefProd rp1;
  }
}

#endif // SIPIXELCLUSTER_CLASSES_H
