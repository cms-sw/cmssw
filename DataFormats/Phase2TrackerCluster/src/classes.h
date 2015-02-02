#ifndef DATAFORMATS_PHASE2TRACKERCLUSTER_CLASSES_H 
#define DATAFORMATS_PHASE2TRACKERCLUSTER_CLASSES_H 

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetNew.h"

namespace {
    struct dictionary {
        edm::Wrapper< Phase2TrackerCluster1D > cl0;
        edm::Wrapper< std::vector< Phase2TrackerCluster1D > > cl1;
        edm::Wrapper< edmNew::DetSet< Phase2TrackerCluster1D > > cl2; 
        edm::Wrapper< std::vector< edmNew::DetSet< Phase2TrackerCluster1D > > > cl3; 
        edm::Wrapper< Phase2TrackerCluster1DCollectionNew > cl4;
    };
}

#endif 
