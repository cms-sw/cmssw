#ifndef DATAFORMATS_PHASE2TRACKERCLUSTER1D_CLASSES_H 
#define DATAFORMATS_PHASE2TRACKERCLUSTER1D_CLASSES_H 

#include "DataFormats/Phase2TrackerCluster1D/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ContainerMask.h"

namespace {
    struct dictionary {
        std::vector< Phase2TrackerCluster1D > v1;
        edm::DetSet< Phase2TrackerCluster1D > ds1; 
        std::vector< edm::DetSet< Phase2TrackerCluster1D > > vds1; 
 
        Phase2TrackerCluster1DCollection c1;
        Phase2TrackerCluster1DCollectionNew c1_new;

        edm::Wrapper< Phase2TrackerCluster1DCollection > w1;
        edm::Wrapper< Phase2TrackerCluster1DCollectionNew > w1_new;

        Phase2TrackerCluster1DRef r1;
        Phase2TrackerCluster1DRefNew r1_new;
 
        Phase2TrackerCluster1DRefVector rv1;
        Phase2TrackerCluster1DRefProd rp1;
 
	edm::Ref< Phase2TrackerCluster1DCollection, edm::DetSet< Phase2TrackerCluster1D >, edm::refhelper::FindDetSetForDetSetVector< Phase2TrackerCluster1D, Phase2TrackerCluster1DCollection > > dsvr_r;
        std::vector< edm::Ref< Phase2TrackerCluster1DCollectionNew, Phase2TrackerCluster1D, Phase2TrackerCluster1DCollectionNew::FindForDetSetVector > > dsvr_v;
        edmNew::DetSetVector< edm::Ref< Phase2TrackerCluster1DCollectionNew, Phase2TrackerCluster1D, Phase2TrackerCluster1DCollectionNew::FindForDetSetVector > > dsvr;
        edm::Wrapper< edmNew::DetSetVector< edm::Ref< Phase2TrackerCluster1DCollectionNew, Phase2TrackerCluster1D, Phase2TrackerCluster1DCollectionNew::FindForDetSetVector > > > dsvr_w;

        edm::ContainerMask< Phase2TrackerCluster1DCollectionNew > cm1;
        edm::Wrapper< edm::ContainerMask< Phase2TrackerCluster1DCollectionNew > > w_cm1;
    };
}

#endif 
