#ifndef DataFormats_Common_ConvertHandle_h
#define DataFormats_Common_ConvertHandle_h

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <typeinfo>

namespace edm {

  namespace handleimpl {
    void throwInvalidReference();
    void throwConvertTypeError(std::type_info const& expected, std::type_info const& actual);
  }

  // Convert from handle-to-void to handle-to-T
  template<typename T>
  void convert_handle(BasicHandle const& bh,
		      Handle<T>& result) {
    if(bh.failedToGet()) {
      Handle<T> h(bh.whyFailed());
      h.swap(result);
      return;
    }
    void const* basicWrapper = bh.wrapper();
    if(basicWrapper == 0) {
      handleimpl::throwInvalidReference();
    }
    if(!(bh.interface()->dynamicTypeInfo() == typeid(T))) {
      handleimpl::throwConvertTypeError(typeid(T), bh.interface()->dynamicTypeInfo());
    }
    Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(basicWrapper);

    Handle<T> h(wrapper->product(), bh.provenance());
    h.swap(result);
  }
}

#endif
