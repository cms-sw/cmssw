#ifndef DATAFORMATS_PHASE2TRACKERCLUSTER1D_CLASSES_H 
#define DATAFORMATS_PHASE2TRACKERCLUSTER1D_CLASSES_H 

#include "DataFormats/Phase2TrackerCluster1D/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
    struct dictionary {
        edm::Wrapper< Phase2TrackerCluster1D > cl0;
        edm::Wrapper< std::vector< Phase2TrackerCluster1D > > cl1;
        edm::Wrapper< edm::DetSet< Phase2TrackerCluster1D > > cl2; 
        edm::Wrapper< std::vector< edm::DetSet< Phase2TrackerCluster1D > > > cl3; 
        edm::Wrapper< Phase2TrackerCluster1DCollection > cl4;
        edm::Wrapper< Phase2TrackerCluster1DCollectionNew > cl5;
    };
}

#endif 
