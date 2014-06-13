#include "DataFormats/Common/interface/WrapperOwningHolder.h"

namespace edm {
  WrapperOwningHolder::EDProductDeleter::EDProductDeleter(WrapperInterfaceBase const* interface) : interface_(interface) {}
  void WrapperOwningHolder::EDProductDeleter::operator()(void const* wrapper) const {
    if(wrapper != 0) {
      interface_->deleteProduct(wrapper);
    }
  }

  WrapperOwningHolder::WrapperOwningHolder() : WrapperHolder(), wrapperOwner_() {}

  WrapperOwningHolder::WrapperOwningHolder(std::shared_ptr<void const> wrapper, WrapperInterfaceBase const* interface) :
      WrapperHolder(wrapper.get(), interface), wrapperOwner_(wrapper) {
  }

  WrapperOwningHolder::WrapperOwningHolder(void const* wrapper, WrapperInterfaceBase const* interface) :
      WrapperHolder(wrapper, interface),
      wrapperOwner_(makeWrapper(wrapper, interface)) {
  }

  std::shared_ptr<void const>
  WrapperOwningHolder::makeWrapper(void const* wrapper, WrapperInterfaceBase const* interface) {
     return(std::shared_ptr<void const>(wrapper, EDProductDeleter(interface)));
  }

}
