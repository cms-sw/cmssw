#ifndef PHASE2TRACKERCLUSTER1D_CLASSES_H
#define PHASE2TRACKERCLUSTER1D_CLASSES_H

#include "DataFormats/Phase2TrackerCluster1D/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include <vector>

namespace {
  struct dictionary {
    edm::Wrapper<Phase2TrackerCluster1D> cl0;
    edm::Wrapper< std::vector<Phase2TrackerCluster1D>  > cl1;
    edm::Wrapper< edm::DetSet<Phase2TrackerCluster1D> > cl2;
    edm::Wrapper< std::vector<edm::DetSet<Phase2TrackerCluster1D> > > cl3;
    edm::Wrapper< edm::DetSetVector<Phase2TrackerCluster1D> > cl4;
    edm::Wrapper<edmNew::DetSetVector<Phase2TrackerCluster1D> > cl4_bis;
  };
}

#endif // PHASE2TRACKERCLUSTER1D_CLASSES_H
