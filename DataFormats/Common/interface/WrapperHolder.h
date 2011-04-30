#ifndef DataFormats_Common_WrapperHolder_h
#define DataFormats_Common_WrapperHolder_h

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Provenance/interface/WrapperInterfaceBase.h"

#include "boost/shared_ptr.hpp"

#include <typeinfo>
#include <vector>

namespace edm {
  class WrapperHolder {
  public:
    enum Ownership {
      Owned,
      NotOwned
    };
    struct EDProductDeleter {
      explicit EDProductDeleter(WrapperInterfaceBase const* interface);
      void operator()(void const* wrapper) const;
      WrapperInterfaceBase const* interface_;
    };

    WrapperHolder();

    WrapperHolder(void const* wrapper, WrapperInterfaceBase const* interface, Ownership ownershipPolicy);

    WrapperHolder(boost::shared_ptr<void const> wrapper, WrapperInterfaceBase const* interface);

    boost::shared_ptr<void const> makeWrapper(void const* wrapper, WrapperInterfaceBase const* interface, Ownership ownershipPolicy);

    bool isValid() const {
      return wrapper() && interface_ != 0;
    }

    void fillView(ProductID const& id,
                  std::vector<void const*>& view,
                  helper_vector_ptr& helpers) const {
      interface_->fillView(wrapper(), id, view, helpers);
    }

    void setPtr(std::type_info const& iToType,
                unsigned long iIndex,
                void const*& oPtr) const {
      interface_->setPtr(wrapper(), iToType, iIndex, oPtr);
    }

    void fillPtrVector(std::type_info const& iToType,
                       std::vector<unsigned long> const& iIndicies,
                       std::vector<void const*>& oPtr) const {
      interface_->fillPtrVector(wrapper(), iToType, iIndicies, oPtr);
    }

    bool isMergeable() const {
      return interface_->isMergeable(wrapper());
    }

    bool hasIsProductEqual() const {
      return interface_->hasIsProductEqual(wrapper());
    }

    bool mergeProduct(void const* newProduct) {
      return interface_->mergeProduct(const_cast<void *>(wrapper()), newProduct);}

    bool isProductEqual(void const* newProduct) const {
      return interface_->isProductEqual(wrapper(), newProduct);
    }

    bool isPresent() const {
      return interface_->isPresent(wrapper());
    }

    std::type_info const& dynamicTypeInfo() const {
      return interface_->dynamicTypeInfo();
    }

    std::type_info const& wrappedTypeInfo() const {
      return interface_->wrappedTypeInfo();
    }

    void const* wrapper() const {
      return wrapper_.get();
    }

    boost::shared_ptr<void const> product() const {
      return wrapper_;
    }

    WrapperInterfaceBase const* interface() const {
      return interface_;
    }

    void reset() {
      interface_ = 0;
      wrapper_.reset();
    }

  private:  
    boost::shared_ptr<void const> wrapper_;
    WrapperInterfaceBase const* interface_;
  };

}
#endif
