#ifndef DataFormats_PixelCluster_classes_h
#define DataFormats_PixelCluster_classes_h
#include "DataFormats/Phase2TrackerCluster2D/interface/Phase2TrackerCluster2D.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

namespace {
  struct dictionary {
    edm::Wrapper<Phase2TrackerCluster2D> cl0;
    edm::Wrapper< std::vector<Phase2TrackerCluster2D>  > cl1;
    edm::Wrapper< edm::DetSet<Phase2TrackerCluster2D> > cl2;
    edm::Wrapper< std::vector<edm::DetSet<Phase2TrackerCluster2D> > > cl3;
    edm::Wrapper< edm::DetSetVector<Phase2TrackerCluster2D> > cl4;
    edm::Wrapper<edmNew::DetSetVector<Phase2TrackerCluster2D> > cl4_bis;
  };
}

#endif // SIPIXELCLUSTER_CLASSES_H
