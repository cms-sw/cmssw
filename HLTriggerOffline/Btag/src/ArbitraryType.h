#ifndef DataFormats_Common_ArbitraryType
#define DataFormats_Common_ArbitraryType

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace edm {
  class   ArbitraryType { };
  typedef Handle<ArbitraryType>  ArbitraryHandle;
  typedef Wrapper<ArbitraryType> ArbitraryWrapper;


// for CMSSW_7_1_X

   template<>
   void convert_handle(BasicHandle && bh,
                       ArbitraryHandle & result) {
     if(bh.failedToGet()) {
       ArbitraryHandle handle(std::move(bh.whyFailedFactory()));
       result = std::move(handle);
       return;
    }
     void const* basicWrapper = bh.wrapper();
     if(basicWrapper == 0) 
         throw Exception(errors::InvalidReference, "NullPointer")      << "edm::BasicHandle has null pointer to Wrapper";
	

     if(!(bh.interface()->dynamicTypeInfo() == typeid(ArbitraryType))) 
	throw Exception(errors::LogicError, "TypeMismatch")
            << "edm::BasicHandle contains a product of type " << bh.interface()->dynamicTypeInfo().name() << ".\n"
         << "A type of " << typeid(ArbitraryType).name() << "was expected.";


     ArbitraryWrapper const* wrapper = static_cast<ArbitraryWrapper const*>(basicWrapper);
     ArbitraryHandle h(wrapper->product(), bh.provenance());
     h.swap(result);
   }



// for CMSSW_7_0_X
/*
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
  
*/


}

#endif // DataFormats_Common_ArbitraryType

