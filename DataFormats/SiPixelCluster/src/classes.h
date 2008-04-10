#ifndef DataFormats_PixelCluster_classes_h
#define DataFormats_PixelCluster_classes_h
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<SiPixelCluster> v1;
    edm::DetSet<SiPixelCluster> ds1;
    std::vector<edm::DetSet<SiPixelCluster> > vds1;
    SiPixelClusterCollection c1;
    SiPixelClusterCollectionNew c1_new;
    edm::Wrapper<SiPixelClusterCollection> w1;
    edm::Wrapper<SiPixelClusterCollectionNew> w1_new;
    SiPixelClusterRef r1;
    SiPixelClusterRefNew r1_new;
    SiPixelClusterRefVector rv1;
    SiPixelClusterRefProd rp1;
  }
}

#endif // SIPIXELCLUSTER_CLASSES_H
