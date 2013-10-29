#ifndef PHASE2TRACKERDIGI_CLASSES_H
#define PHASE2TRACKERDIGI_CLASSES_H

#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include <vector>

namespace {
  struct dictionary {
    
    edm::Wrapper<Phase2TrackerDigi> zs0;
    edm::Wrapper< std::vector<Phase2TrackerDigi>  > zs1;
    edm::Wrapper< edm::DetSet<Phase2TrackerDigi> > zs2;
    edm::Wrapper< std::vector<edm::DetSet<Phase2TrackerDigi> > > zs3;
    edm::Wrapper< edm::DetSetVector<Phase2TrackerDigi> > zs4;
    edm::Wrapper<edmNew::DetSetVector<Phase2TrackerDigi> > zs4_bis;
    
  };
}

#endif // PHASE2TRACKERDIGI_CLASSES_H
