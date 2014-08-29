#ifndef DataFormats_Common_ConvertHandle_h
#define DataFormats_Common_ConvertHandle_h

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
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
  void convert_handle(BasicHandle && bh,
		      Handle<T>& result) {
    if(bh.failedToGet()) {
      Handle<T> h(std::move(bh.whyFailedFactory()));
      result = std::move(h);
      return;
    }
    void const* basicWrapper = bh.wrapper();
    if(basicWrapper == 0) {
      handleimpl::throwInvalidReference();
    }
    if(!(bh.wrapper()->dynamicTypeInfo() == typeid(T))) {
      handleimpl::throwConvertTypeError(typeid(T), bh.wrapper()->dynamicTypeInfo());
    }
    Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(basicWrapper);

    Handle<T> h(wrapper->product(), bh.provenance());
    h.swap(result);
  }
}
#else
namespace edm {
  class BasicHandle;
  template<typename T> class Handle;
  
  template<typename T>
  void convert_handle(BasicHandle & bh,
                      Handle<T>& result);
}
#endif

#endif
