#ifndef DataFormats_Phase2ITPixelCluster_classes_h
#define DataFormats_Phase2ITPixelCluster_classes_h
#include "DataFormats/Phase2ITPixelCluster/interface/Phase2PixelCluster.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/Phase2ITPixelCluster/interface/Phase2PixelClusterShapeCache.h"

namespace DataFormats_Phase2PixelCluster {
  struct dictionary {
    std::vector<Phase2PixelCluster> v1;
    edm::DetSet<Phase2PixelCluster> ds1;
    std::vector<edm::DetSet<Phase2PixelCluster> > vds1;
    Phase2PixelClusterCollection c1;
    Phase2PixelClusterCollectionNew c1_new;
    edm::Wrapper<Phase2PixelClusterCollection> w1;
    edm::Wrapper<Phase2PixelClusterCollectionNew> w1_new;
    Phase2PixelClusterRef r1;
    Phase2PixelClusterRefNew r1_new;
    Phase2PixelClusterRefVector rv1;
    Phase2PixelClusterRefProd rp1;
    edm::Ref<edm::DetSetVector<Phase2PixelCluster>,edm::DetSet<Phase2PixelCluster>,edm::refhelper::FindDetSetForDetSetVector<Phase2PixelCluster,edm::DetSetVector<Phase2PixelCluster> > > boguscrap;

    std::vector<edm::Ref<edmNew::DetSetVector<Phase2PixelCluster>,Phase2PixelCluster,edmNew::DetSetVector<Phase2PixelCluster>::FindForDetSetVector> > dsvr_v;
    edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<Phase2PixelCluster>,Phase2PixelCluster,edmNew::DetSetVector<Phase2PixelCluster>::FindForDetSetVector> > dsvr;
    edm::Wrapper<edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<Phase2PixelCluster>,Phase2PixelCluster,edmNew::DetSetVector<Phase2PixelCluster>::FindForDetSetVector> > > dsvr_w;

    edm::ContainerMask<Phase2PixelClusterCollectionNew> cm1;
    edm::Wrapper<edm::ContainerMask<Phase2PixelClusterCollectionNew> > w_cm1;

    Phase2PixelClusterShapeCache clusterShapeCache;
    edm::Wrapper<Phase2PixelClusterShapeCache> wclusterShapeCache;
  };
}

#endif // PHASE2PIXELCLUSTER_CLASSES_H
