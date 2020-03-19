#ifndef FWCore_Utilities_ESProductTag_h
#define FWCore_Utilities_ESProductTag_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     ESProductTag
//
/**\class ESProductTag ESProductTag.h "ESProductTag.h"

 Description: Contains all information to uniquely identify a product in the EventSetup system

 Usage:
    The strings used to initialize an ESProductTag are the same as those for an ESInputTag.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 18 Sep 2019 16:01:26 GMT
//

// system include files

// user include files
#include "FWCore/Utilities/interface/ESInputTag.h"

// forward declarations
namespace edm {
  template <typename ESProduct, typename ESRecord>
  class ESProductTag {
  public:
    using Type = ESProduct;
    using Record = ESRecord;

    ESProductTag(std::string iModuleLabel, std::string iDataLabel)
        : tag_{std::move(iModuleLabel), std::move(iDataLabel)} {}

    ESProductTag(ESInputTag iTag) : tag_{std::move(iTag)} {}

    // ---------- const member functions ---------------------
    ESInputTag const& inputTag() const noexcept { return tag_; }
    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

  private:
    // ---------- member data --------------------------------
    ESInputTag tag_;
  };
}  // namespace edm

#endif
