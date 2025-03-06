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
  class SignallingProductRegistry : public ProductRegistry {
  public:
    SignallingProductRegistry() : ProductRegistry(), productAddedSignal_(), typeAddedStack_() {}

    explicit SignallingProductRegistry(ProductRegistry const& preg)
        : ProductRegistry(preg.productList(), false), productAddedSignal_(), typeAddedStack_() {}
    signalslot::Signal<void(ProductDescription const&)> productAddedSignal_;

    struct Copy {};

    SignallingProductRegistry(ProductRegistry const& preg, Copy): ProductRegistry(preg), productAddedSignal_(), typeAddedStack_() {};
    
    void addProduct(ProductDescription const& productdesc, bool iFromListener = false) {
      addProduct_(productdesc, iFromListener);
    }

    void addLabelAlias(ProductDescription const& productdesc,
                       std::string const& labelAlias,
                       std::string const& instanceAlias) {
      addLabelAlias_(productdesc, labelAlias, instanceAlias);
    }

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

    ProductRegistry const& registry() { return *this; }

  private:
    void addCalled(ProductDescription const&, bool) override;
    // ---------- member data --------------------------------
    std::map<std::string, unsigned int> typeAddedStack_;
  };
}  // namespace edm

#endif
