#ifndef DataFormats_Common_ArbitraryType
#define DataFormats_Common_ArbitraryType

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace edm {
  class   ArbitraryType { };
  typedef Handle<ArbitraryType>  ArbitraryHandle;
  typedef Wrapper<ArbitraryType> ArbitraryWrapper;

  template <>
  void convert_handle(const BasicHandle & basic, ArbitraryHandle & result)
  {
    if (basic.failedToGet()) {
      ArbitraryHandle handle(basic.whyFailed());
      handle.swap(result);
      return;
    }
    
    const ArbitraryWrapper * wrapper = static_cast<const ArbitraryWrapper *>(basic.wrapper());
    if (wrapper == 0 or wrapper->product() == 0) {
      boost::shared_ptr<cms::Exception> whyFailed( new edm::Exception(edm::errors::ProductNotFound, "InvalidID") );
      *whyFailed << "get by product ID: no product with given id: " << basic.id() << "\n";
      ArbitraryHandle handle(whyFailed);
      handle.swap(result);
      return;
    }
    
    ArbitraryHandle handle(wrapper->product(), basic.provenance());
    handle.swap(result);
  }
  
}

#endif // DataFormats_Common_ArbitraryType
