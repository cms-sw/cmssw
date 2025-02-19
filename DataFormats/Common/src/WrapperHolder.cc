#include "DataFormats/Common/interface/WrapperHolder.h"
#include <iostream>

namespace edm {
  WrapperHolder::WrapperHolder() : wrapper_(), interface_(0) {}

  WrapperHolder::WrapperHolder(void const* wrapper, WrapperInterfaceBase const* interface) :
      wrapper_(wrapper),
      interface_(interface) {
  }
}
