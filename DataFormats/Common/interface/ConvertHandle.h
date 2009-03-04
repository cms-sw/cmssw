#ifndef DataFormats_Common_ConvertHandle_h
#define DataFormats_Common_ConvertHandle_h

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace edm {

  namespace handleimpl {
    void throwInvalidReference();
    void throwConvertTypeError(EDProduct const* product);
  }

  // Convert from handle-to-EDProduct to handle-to-T
  template <class T>
  void convert_handle(BasicHandle const& orig,
		      Handle<T>& result) {
    if(orig.failedToGet()) {
      Handle<T> h(orig.whyFailed());
      h.swap(result);
      return;
    }
    EDProduct const* originalWrap = orig.wrapper();
    if (originalWrap == 0) {
      handleimpl::throwInvalidReference();
    }
    Wrapper<T> const* wrap = dynamic_cast<Wrapper<T> const*>(originalWrap);
    if (wrap == 0) {
      handleimpl::throwConvertTypeError(originalWrap);
    }

    Handle<T> h(wrap->product(), orig.provenance());
    h.swap(result);
  }
}

#endif
