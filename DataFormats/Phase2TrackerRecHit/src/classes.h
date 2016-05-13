#ifndef DATAFORMATS_PHASE2TRACKERRECHIT_CLASSES_H 
#define DATAFORMATS_PHASE2TRACKERRECHIT_CLASSES_H 

#include "DataFormats/Phase2TrackerRecHit/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetNew.h"

namespace {
    struct dictionary_ph2rh {
        edm::Wrapper< Phase2TrackerRecHit1D > cl0;
        edm::Wrapper< std::vector< Phase2TrackerRecHit1D > > cl1;
        edm::Wrapper< edmNew::DetSet< Phase2TrackerRecHit1D > > cl2; 
        edm::Wrapper< std::vector< edmNew::DetSet< Phase2TrackerRecHit1D > > > cl3; 
        edm::Wrapper< Phase2TrackerRecHit1DCollectionNew > cl4;
    };
}

#endif 
