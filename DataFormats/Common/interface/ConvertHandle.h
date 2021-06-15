#ifndef DataFormats_Common_ConvertHandle_h
#define DataFormats_Common_ConvertHandle_h

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/Likely.h"

#include <typeinfo>
#include <algorithm>
#include <memory>

namespace edm {

  namespace handleimpl {
    void throwConvertTypeError(std::type_info const& expected, std::type_info const& actual);
    std::shared_ptr<edm::HandleExceptionFactory const> makeInvalidReferenceException();
  }  // namespace handleimpl

  // Convert from handle-to-void to handle-to-T
  template <typename T>
  Handle<T> convert_handle(BasicHandle&& bh) noexcept(true) {
    if UNLIKELY (bh.failedToGet()) {
      return Handle<T>(std::move(bh.whyFailedFactory()));
    }
    void const* basicWrapper = bh.wrapper();
    if UNLIKELY (nullptr == basicWrapper) {
      return Handle<T>{handleimpl::makeInvalidReferenceException()};
    }
    auto wrapper = static_cast<Wrapper<T> const*>(basicWrapper);

    return Handle<T>(wrapper->product(), bh.provenance());
  }

  template <typename T>
  Handle<T> convert_handle_check_type(BasicHandle&& bh) {
    if UNLIKELY (bh.failedToGet()) {
      return Handle<T>(std::move(bh.whyFailedFactory()));
    }
    void const* basicWrapper = bh.wrapper();
    if UNLIKELY (basicWrapper == nullptr) {
      return Handle<T>{handleimpl::makeInvalidReferenceException()};
    }
    if UNLIKELY (!(bh.wrapper()->dynamicTypeInfo() == typeid(T))) {
      handleimpl::throwConvertTypeError(typeid(T), bh.wrapper()->dynamicTypeInfo());
    }
    Wrapper<T> const* wrapper = static_cast<Wrapper<T> const*>(basicWrapper);

    return Handle<T>(wrapper->product(), bh.provenance());
  }

}  // namespace edm

#endif
