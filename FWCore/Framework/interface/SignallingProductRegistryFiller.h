#ifndef FWCore_Framework_SignallingProductRegistryFiller_h
#define FWCore_Framework_SignallingProductRegistryFiller_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     SignallingProductRegistryFiller
//
/**\class SignallingProductRegistryFiller SignallingProductRegistryFiller.h FWCore/Framework/interface/SignallingProductRegistryFiller.h

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
  class SignallingProductRegistryFiller {
  public:
    SignallingProductRegistryFiller() : productAddedSignal_(), registry_(), typeAddedStack_() {}

    explicit SignallingProductRegistryFiller(ProductRegistry const& preg)
        : productAddedSignal_(), registry_(preg), typeAddedStack_() {};

    SignallingProductRegistryFiller(SignallingProductRegistryFiller const&) = delete;
    SignallingProductRegistryFiller(SignallingProductRegistryFiller&&) = delete;
    SignallingProductRegistryFiller& operator=(SignallingProductRegistryFiller const&) = delete;
    SignallingProductRegistryFiller& operator=(SignallingProductRegistryFiller&&) = delete;

    signalslot::Signal<void(ProductDescription const&)> productAddedSignal_;

    void addProduct(ProductDescription const& productdesc, bool iFromListener = false) {
      registry_.addProduct_(productdesc);
      addCalled(productdesc, iFromListener);
    }

    void addLabelAlias(ProductDescription const& productdesc,
                       std::string const& labelAlias,
                       std::string const& instanceAlias) {
      addCalled(registry_.addLabelAlias_(productdesc, labelAlias, instanceAlias), false);
    }
    void addFromInput(edm::ProductRegistry const& iReg) {
      registry_.addFromInput_(iReg, [this](auto const& prod) { addCalled(prod, false); });
    }

    //NOTE: this is not const since we only want items that have non-const access to this class to be
    // able to call this internal iteration
    // Called only for branches that are present (part of avoiding creating type information for dropped branches)
    template <typename T>
    void callForEachBranch(T const& iFunc) {
      //NOTE: If implementation changes from a map, need to check that iterators are still valid
      // after an insert with the new container, else need to copy the container and iterate over the copy
      for (auto const& entry : registry_.productList()) {
        if (entry.second.present()) {
          iFunc(entry.second);
        }
      }
    }
    void setUnscheduledProducts(std::set<std::string> const& unscheduledLabels) {
      registry_.setUnscheduledProducts(unscheduledLabels);
    }
    ProductRegistry::ProductList& productListUpdator() { return registry_.productListUpdator(); }

    template <class T>
    void watchProductAdditions(const T& iFunc) {
      serviceregistry::connect_but_block_self(productAddedSignal_, iFunc);
    }
    template <class T, class TMethod>
    void watchProductAdditions(T const& iObj, TMethod iMethod) {
      using std::placeholders::_1;
      serviceregistry::connect_but_block_self(productAddedSignal_, std::bind(iMethod, iObj, _1));
    }

    ProductRegistry moveTo() { return std::move(registry_); }
    ProductRegistry const& registry() const { return registry_; }

    void setFrozen(bool initializeLookupInfo = true) { registry_.setFrozen(initializeLookupInfo); }

    void setFrozen(std::set<TypeID> const& productTypesConsumed,
                   std::set<TypeID> const& elementTypesConsumed,
                   std::string const& processName) {
      registry_.setFrozen(productTypesConsumed, elementTypesConsumed, processName);
    }

  private:
    void addCalled(ProductDescription const&, bool);
    // ---------- member data --------------------------------
    ProductRegistry registry_;
    std::map<std::string, unsigned int> typeAddedStack_;
  };
}  // namespace edm

#endif
