#ifndef SiPixelCluster_SiPixelClusterFwd_h
#define SiPixelCluster_SiPixelClusterFwd_h

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

typedef edm::DetSetVector<SiPixelCluster> SiPixelClusterCollection;
typedef edm::Ref<SiPixelClusterCollection, SiPixelCluster> SiPixelClusterRef;
//warning: in the following the default does not work!
//typedef edm::RefVector<edm::DetSetVector<SiPixelCluster>,SiPixelCluster,edm::refhelper::FindForDetSetVector<SiPixelCluster> > SiPixelClusterRefVector;
typedef edm::RefProd<SiPixelClusterCollection> SiPixelClusterRefProd;

#endif
