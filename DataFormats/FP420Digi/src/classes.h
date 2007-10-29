#ifndef DataFormats_HDigiFP420_CLASSES_H
#define DataFormats_HDigiFP420_CLASSES_H

//#include "SimRomanPot/SimFP420/interface/HDigiFP420.h"
//#include "SimRomanPot/SimFP420/interface/DigiCollectionFP420.h"

//#include "SimRomanPot/DataFormats/interface/HDigiFP420.h"
//#include "SimRomanPot/DataFormats/interface/DigiCollectionFP420.h"

#include "DataFormats/FP420Digi/interface/HDigiFP420.h"
#include "DataFormats/FP420Digi/interface/DigiCollectionFP420.h"
//#include "boost/cstdint.hpp" 
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include <vector>

namespace {
  namespace {
    edm::Wrapper<HDigiFP420 > zs0;
    edm::Wrapper<DigiCollectionFP420> zsc0;
    edm::Wrapper<std::vector<HDigiFP420> > zs1;
    edm::Wrapper< edm::DetSet<HDigiFP420> > zs2;
    edm::Wrapper< std::vector<edm::DetSet<HDigiFP420> > > zs3;
    edm::Wrapper< edm::DetSetVector<HDigiFP420> > zs4;
  }
}
#endif // HDigiFP420_CLASSES_H
