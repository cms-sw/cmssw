#ifndef FWCore_Framework_ProducesCollector_h
#define FWCore_Framework_ProducesCollector_h

// Package:     FWCore/Framework
// Class  :     edm::ProducesCollector
//
/**\class edm::ProducesCollector

 Description: Helper class to gather produces information for the ProducerBase class.

 Usage:
    The constructor of a module can get an instance of edm::ProducesCollector by calling its
producesCollector() method. This instance can then be passed to helper classes in order to register
the data the helper will put into the Event, LuminosityBlock or Run.

    Note this only supports use of the Transition enum. The produce template functions using
the BranchType enum are not supported by this class. We still support the BranchType enum in
ProducerBase for backward compatibility reasons, but even there they are deprecated and we
are trying to migrate client code to use the template functions using the Transition enum.

     WARNING: The ProducesCollector should be used during the time that modules are being
constructed. It should not be saved and used later. It will not work if it is used to call
the produces function during beginJob, beginRun, beginLuminosity block, event processing or
at any later time. It can be used while the module constructor is running or be contained in
a functor passed to the Framework with a call to callWhenNewProductsRegistered.

*/
//
// Original Author:  W. David Dagenhart
//         Created:  24 September 2019

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Transition.h"

#include <string>
#include <utility>

namespace edm {

  class TypeID;

  class ProducesCollector {
  public:
    ProducesCollector() = delete;
    ProducesCollector(ProducesCollector const&);
    ProducesCollector(ProducesCollector&&) = default;
    ProducesCollector& operator=(ProducesCollector const&);
    ProducesCollector& operator=(ProducesCollector&&) = default;

    template <class ProductType>
    ProductRegistryHelper::BranchAliasSetterT<ProductType> produces() {
      return helper_->produces<ProductType>();
    }

    template <class ProductType>
    ProductRegistryHelper::BranchAliasSetterT<ProductType> produces(std::string instanceName) {
      return helper_->produces<ProductType>(std::move(instanceName));
    }

    template <typename ProductType, Transition B>
    ProductRegistryHelper::BranchAliasSetterT<ProductType> produces() {
      return helper_->produces<ProductType, B>();
    }

    template <typename ProductType, Transition B>
    ProductRegistryHelper::BranchAliasSetterT<ProductType> produces(std::string instanceName) {
      return helper_->produces<ProductType, B>(std::move(instanceName));
    }

    ProductRegistryHelper::BranchAliasSetter produces(const TypeID& id,
                                                      std::string instanceName = std::string(),
                                                      bool recordProvenance = true);

    template <Transition B>
    ProductRegistryHelper::BranchAliasSetter produces(const TypeID& id,
                                                      std::string instanceName = std::string(),
                                                      bool recordProvenance = true) {
      return helper_->produces<B>(id, std::move(instanceName), recordProvenance);
    }

  private:
    friend class ProducerBase;
    ProducesCollector(ProductRegistryHelper* helper);

    propagate_const<ProductRegistryHelper*> helper_;
  };

}  // namespace edm
#endif
