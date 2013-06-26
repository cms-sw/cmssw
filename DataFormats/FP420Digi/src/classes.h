#ifndef DataFormats_HDigiFP420_CLASSES_H
#define DataFormats_HDigiFP420_CLASSES_H

#include "DataFormats/FP420Digi/interface/HDigiFP420.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
#include <vector>
#include <string>

namespace {
  struct dictionary {
    edm::Wrapper<HDigiFP420 > zs0;
    edm::Wrapper<std::vector<HDigiFP420> > zs1;
    edm::Wrapper< edm::DetSet<HDigiFP420> > zs2;
    edm::Wrapper< std::vector<edm::DetSet<HDigiFP420> > > zs3;
    edm::Wrapper< edm::DetSetVector<HDigiFP420> > zs4;

    edm::Wrapper<DigiCollectionFP420> collection;
  };
}

#endif // HDigiFP420_CLASSES_H
