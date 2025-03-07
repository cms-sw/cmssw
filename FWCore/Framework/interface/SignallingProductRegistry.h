#ifndef FWCore_Framework_SignallingProductRegistry_h
#define FWCore_Framework_SignallingProductRegistry_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     SignallingProductRegistry
//
/**\class SignallingProductRegistry SignallingProductRegistry.h FWCore/Framework/interface/SignallingProductRegistry.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 23 16:47:10 CEST 2005
//

// system include files
#include <map>
#include <string>

#include "FWCore/Utilities/interface/Signal.h"

// user include files
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/connect_but_block_self.h"

// forward declarations
namespace edm {
  class SignallingProductRegistry : private ProductRegistry {
  public:
    SignallingProductRegistry() : ProductRegistry(), productAddedSignal_(), typeAddedStack_() {}

    explicit SignallingProductRegistry(ProductRegistry const& preg)
        : ProductRegistry(preg.productList(), false), productAddedSignal_(), typeAddedStack_() {}
    signalslot::Signal<void(ProductDescription const&)> productAddedSignal_;

    struct Copy {};

    SignallingProductRegistry(ProductRegistry const& preg, Copy)
        : ProductRegistry(preg), productAddedSignal_(), typeAddedStack_() {};

    void addProduct(ProductDescription const& productdesc, bool iFromListener = false) {
      addProduct_(productdesc, iFromListener);
    }

    void addLabelAlias(ProductDescription const& productdesc,
                       std::string const& labelAlias,
                       std::string const& instanceAlias) {
      addLabelAlias_(productdesc, labelAlias, instanceAlias);
    }
    void addFromInput(edm::ProductRegistry const& iReg) { addFromInput_(iReg); }

    //NOTE: this is not const since we only want items that have non-const access to this class to be
    // able to call this internal iteration
    // Called only for branches that are present (part of avoiding creating type information for dropped branches)
    template <typename T>
    void callForEachBranch(T const& iFunc) {
      //NOTE: If implementation changes from a map, need to check that iterators are still valid
      // after an insert with the new container, else need to copy the container and iterate over the copy
      for (ProductRegistry::ProductList::const_iterator itEntry = productList().begin(), itEntryEnd = productList().end();
           itEntry != itEntryEnd;
           ++itEntry) {
        if (itEntry->second.present()) {
          iFunc(itEntry->second);
        }
      }
    }
    void setUnscheduledProducts(std::set<std::string> const& unscheduledLabels) {
      ProductRegistry::setUnscheduledProducts(unscheduledLabels);
    }
    ProductList& productListUpdator() { return ProductRegistry::productListUpdator(); }

    SignallingProductRegistry(SignallingProductRegistry const&) = delete;             // Disallow copying and moving
    SignallingProductRegistry& operator=(SignallingProductRegistry const&) = delete;  // Disallow copying and moving

    template <class T>
    void watchProductAdditions(const T& iFunc) {
      serviceregistry::connect_but_block_self(productAddedSignal_, iFunc);
    }
    template <class T, class TMethod>
    void watchProductAdditions(T const& iObj, TMethod iMethod) {
      using std::placeholders::_1;
      serviceregistry::connect_but_block_self(productAddedSignal_, std::bind(iMethod, iObj, _1));
    }

    ProductRegistry moveTo() { return std::move(*this); }
    ProductRegistry const& registry() const { return *this; }

    void setFrozen(bool initializeLookupInfo = true) { ProductRegistry::setFrozen(initializeLookupInfo); }

    void setFrozen(std::set<TypeID> const& productTypesConsumed,
                   std::set<TypeID> const& elementTypesConsumed,
                   std::string const& processName) {
      ProductRegistry::setFrozen(productTypesConsumed, elementTypesConsumed, processName);
    }

  private:
    void addCalled(ProductDescription const&, bool) override;
    // ---------- member data --------------------------------
    std::map<std::string, unsigned int> typeAddedStack_;
  };
}  // namespace edm

#endif
