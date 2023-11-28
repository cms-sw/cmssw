// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESTagGetter
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu, 26 Sep 2019 18:34:49 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESTagGetter.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

using namespace edm;

//
// const member functions
//
ESResolverIndex ESTagGetter::operator()(std::string_view iModuleLabel, std::string_view iProductLabel) const {
  ESResolverIndex returnValue = eventsetup::ESRecordsToProductResolverIndices::missingResolverIndex();
  for (auto const& item : lookup_) {
    if (item.productLabel_ == iProductLabel) {
      if (iProductLabel.empty() or iProductLabel == item.productLabel_) {
        return item.index_;
      }
      break;
    }
  }
  return returnValue;
}
