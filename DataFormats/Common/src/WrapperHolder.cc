#include "DataFormats/Common/interface/WrapperHolder.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include <iostream>

namespace edm {
  WrapperHolder::EDProductDeleter::EDProductDeleter(WrapperInterfaceBase const* interface) : interface_(interface) {}
  void WrapperHolder::EDProductDeleter::operator()(void const* wrapper) const {
    if(wrapper != 0) {
      interface_->deleteProduct(wrapper);
    }
  }

  WrapperHolder::WrapperHolder() : wrapper_(), interface_(0) {}

  WrapperHolder::WrapperHolder(boost::shared_ptr<void const> wrapper, WrapperInterfaceBase const* interface) :
      wrapper_(wrapper),
      interface_(interface) {
  }

  WrapperHolder::WrapperHolder(void const* wrapper, WrapperInterfaceBase const* interface, Ownership OwnershipPolicy) :
      wrapper_(makeWrapper(wrapper, interface, OwnershipPolicy)),
      interface_(interface) {
  }

  boost::shared_ptr<void const>
  WrapperHolder::makeWrapper(void const* wrapper, WrapperInterfaceBase const* interface, Ownership OwnershipPolicy) {
     return(OwnershipPolicy == Owned ? boost::shared_ptr<void const>(wrapper, EDProductDeleter(interface)) : boost::shared_ptr<void const>(wrapper, do_nothing_deleter()));
  }
}
