#ifndef FWCore_Framework_ProductResolversFactory_h
#define FWCore_Framework_ProductResolversFactory_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ProductResolversFactory
//
/**\class edm::ProductResolversFactory ProductResolversFactory.h "ProductResolversFactory.h"

 Description: Creates ProductResolvers

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 30 Dec 2024
//

// system include files
#include <memory>
#include "DataFormats/Provenance/interface/ProductRegistry.h"

// user include files

// forward declarations
namespace edm {

  class ProductResolverBase;
  class ProductResolverIndexHelper;

  template <typename F>
  concept ProductResolversFactory =
      requires(F&& f, std::string const& name, ProductRegistry const& reg) { f(InEvent, name, reg); };

  namespace productResolversFactory {
    std::vector<std::shared_ptr<ProductResolverBase>> make(BranchType bt,
                                                           std::string_view iProcessName,
                                                           ProductRegistry const& iReg);
    inline std::vector<std::shared_ptr<ProductResolverBase>> makePrimary(BranchType bt,
                                                                         std::string_view iProcessName,
                                                                         ProductRegistry const& iReg) {
      return make(bt, iProcessName, iReg);
    }

  };  // namespace productResolversFactory
}  // namespace edm

#endif
