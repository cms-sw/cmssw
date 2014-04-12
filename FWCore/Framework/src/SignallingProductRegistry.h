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

// forward declarations
namespace edm {
   class SignallingProductRegistry : public ProductRegistry {

   public:
      SignallingProductRegistry() : ProductRegistry(), productAddedSignal_(), typeAddedStack_() {}
      explicit SignallingProductRegistry(ProductRegistry const& preg) : ProductRegistry(preg.productList(), false), productAddedSignal_(), typeAddedStack_() {}
      signalslot::Signal<void(BranchDescription const&)> productAddedSignal_;

      SignallingProductRegistry(SignallingProductRegistry const&) = delete; // Disallow copying and moving
      SignallingProductRegistry& operator=(SignallingProductRegistry const&) = delete; // Disallow copying and moving

   private:
      virtual void addCalled(BranchDescription const&, bool);
      // ---------- member data --------------------------------
      std::map<std::string, unsigned int> typeAddedStack_;
   };
}

#endif
