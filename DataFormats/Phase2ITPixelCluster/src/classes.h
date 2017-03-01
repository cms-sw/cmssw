#ifndef DataFormats_Phase2ITPixelCluster_classes_h
#define DataFormats_Phase2ITPixelCluster_classes_h
#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelCluster.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelClusterShapeCache.h"

namespace DataFormats_Phase2ITPixelCluster {
  struct dictionary {
    std::vector<Phase2ITPixelCluster> v1;
    edm::DetSet<Phase2ITPixelCluster> ds1;
    std::vector<edm::DetSet<Phase2ITPixelCluster> > vds1;
    Phase2ITPixelClusterCollection c1;
    Phase2ITPixelClusterCollectionNew c1_new;
    edm::Wrapper<Phase2ITPixelClusterCollection> w1;
    edm::Wrapper<Phase2ITPixelClusterCollectionNew> w1_new;
    Phase2ITPixelClusterRef r1;
    Phase2ITPixelClusterRefNew r1_new;
    Phase2ITPixelClusterRefVector rv1;
    Phase2ITPixelClusterRefProd rp1;
    edm::Ref<edm::DetSetVector<Phase2ITPixelCluster>,edm::DetSet<Phase2ITPixelCluster>,edm::refhelper::FindDetSetForDetSetVector<Phase2ITPixelCluster,edm::DetSetVector<Phase2ITPixelCluster> > > boguscrap;

    std::vector<edm::Ref<edmNew::DetSetVector<Phase2ITPixelCluster>,Phase2ITPixelCluster,edmNew::DetSetVector<Phase2ITPixelCluster>::FindForDetSetVector> > dsvr_v;
    edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<Phase2ITPixelCluster>,Phase2ITPixelCluster,edmNew::DetSetVector<Phase2ITPixelCluster>::FindForDetSetVector> > dsvr;
    edm::Wrapper<edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<Phase2ITPixelCluster>,Phase2ITPixelCluster,edmNew::DetSetVector<Phase2ITPixelCluster>::FindForDetSetVector> > > dsvr_w;

    edm::ContainerMask<Phase2ITPixelClusterCollectionNew> cm1;
    edm::Wrapper<edm::ContainerMask<Phase2ITPixelClusterCollectionNew> > w_cm1;

    Phase2ITPixelClusterShapeCache clusterShapeCache;
    edm::Wrapper<Phase2ITPixelClusterShapeCache> wclusterShapeCache;
  };
}

#endif // PHASE2PIXELCLUSTER_CLASSES_H
