#ifndef PHASE2TRACKERSTUB_CLASSES_H
#define PHASE2TRACKERSTUB_CLASSES_H

#include "DataFormats/Phase2TrackerStub/interface/Phase2TrackerStub.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include <vector>

namespace {
  struct dictionary {
    edm::Wrapper<Phase2TrackerStub> cl0;
    edm::Wrapper< std::vector<Phase2TrackerStub>  > cl1;
    edm::Wrapper< edm::DetSet<Phase2TrackerStub> > cl2;
    edm::Wrapper< std::vector<edm::DetSet<Phase2TrackerStub> > > cl3;
    edm::Wrapper< edm::DetSetVector<Phase2TrackerStub> > cl4;
    edm::Wrapper<edmNew::DetSetVector<Phase2TrackerStub> > cl4_bis;
  };
}

#endif // PHASE2TRACKERSTUB_CLASSES_H
