// CMSSW include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/TypeID.h"

namespace edm {

  // Specialize the convert_handle free function for Handle<WrapperBase>
  void convert_handle(BasicHandle&& handle, Handle<WrapperBase>& result) {
    if (handle.failedToGet()) {
      result.setWhyFailedFactory(handle.whyFailedFactory());
      return;
    }

    WrapperBase const* wrapper = handle.wrapper();
    if (wrapper == nullptr) {
      throw Exception(errors::InvalidReference, "NullPointer") << "edm::BasicHandle has null pointer to Wrapper";
    }

    if (TypeID(wrapper->dynamicTypeInfo()) != result.type()) {
      throw Exception(errors::LogicError) << "WrapperBase asked for " << typeDemangle(result.type().name())
                                          << " but was given a " << typeDemangle(wrapper->dynamicTypeInfo().name());
    }

    // Move the handle into result
    result = Handle<WrapperBase>(wrapper, handle.provenance());
  }

  // Specialize the Event::getByToken method for Handle<WrapperBase>
  template <>
  bool Event::getByToken<WrapperBase>(EDGetToken token, Handle<WrapperBase>& result) const {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(result.type(), PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (UNLIKELY(result.failedToGet())) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

}  // namespace edm
