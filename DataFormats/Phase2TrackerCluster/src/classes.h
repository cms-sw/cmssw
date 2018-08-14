#ifndef DATAFORMATS_PHASE2TRACKERCLUSTER_CLASSES_H 
#define DATAFORMATS_PHASE2TRACKERCLUSTER_CLASSES_H 

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetNew.h"

namespace DataFormats_Phase2TrackerCluster {
    struct dictionary_ph2cl {
        edm::Wrapper< Phase2TrackerCluster1D > cl0;
        edm::Wrapper< std::vector< Phase2TrackerCluster1D > > cl1;
        edm::Wrapper< edmNew::DetSet< Phase2TrackerCluster1D > > cl2; 
        edm::Wrapper< std::vector< edmNew::DetSet< Phase2TrackerCluster1D > > > cl3; 
        edm::Wrapper< Phase2TrackerCluster1DCollectionNew > cl4;
        edm::Wrapper< edm::Ref< Phase2TrackerCluster1DCollectionNew, Phase2TrackerCluster1D > > cl5;

        edm::ContainerMask<Phase2TrackerCluster1DCollectionNew > cm1;
        edm::RefProd<edmNew::DetSetVector<Phase2TrackerCluster1D> > cm2;
        edm::Wrapper<edm::ContainerMask<Phase2TrackerCluster1DCollectionNew> > w_cm1;
    };
}

#endif 
