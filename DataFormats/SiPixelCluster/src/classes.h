#ifndef DataFormats_PixelCluster_classes_h
#define DataFormats_PixelCluster_classes_h

#include "DataFormats/SiPixelCluster/interface/SiPixelClusterCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    SiPixelClusterCollection a;
    SiPixelCluster b;
    SiPixelCluster::Pixel c;
    SiPixelCluster::PixelPos d;
    SiPixelCluster::Shift e;
    edm::Wrapper<SiPixelClusterCollection> siPixelClusterCollectionWrapper;
  }
}

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
namespace {
  namespace {
    edm::Wrapper<SiPixelCluster> dummy0;
    edm::Wrapper< std::vector<SiPixelCluster>  > dummy1;
    edm::Wrapper< edm::DetSet<SiPixelCluster> > dummy2;
    edm::Wrapper< std::vector<edm::DetSet<SiPixelCluster> > > dummy3;
    edm::Wrapper< edm::DetSetVector<SiPixelCluster> > dummy4;
  }
}

#endif // SISTRIPCLUSTER_CLASSES_H
