#ifndef WrapperHandle_h
#define WrapperHandle_h

#include <cassert>
#include <memory>
#include <string>
#include <typeinfo>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/TypeID.h"


namespace edm {

  // specialise Handle for a WrapperBase
    
  template <>
  class Handle<WrapperBase> : public HandleBase {
  public:
    // default constructor
    Handle() :
      HandleBase(),
      type_(nullptr)
    { }

    // throws exception if `type` is not a known C++ class type
    Handle(std::string const& type) :
      HandleBase(),
      type_(& TypeWithDict::byName(type).typeInfo())
    { }

    Handle(WrapperBase const* wrapper, Provenance const* prov, std::string const& type) :
      HandleBase(wrapper, prov),
      type_(& TypeWithDict::byName(type).typeInfo())
    { }

    // throws exception if `type` is invalid
    Handle(std::type_info const& type) :
      HandleBase(),
      type_(& type)
    { }

    Handle(WrapperBase const* wrapper, Provenance const* prov, std::type_info const& type) :
      HandleBase(wrapper, prov),
      type_(& type)
    { }

    // used when the attempt to get the data failed
    Handle(std::shared_ptr<HandleExceptionFactory>&& whyFailed) :
      HandleBase(std::move(whyFailed)),
      type_(nullptr)
    { }

    // copiable and moveable
    Handle(Handle<WrapperBase> const& h) = default;
    Handle(Handle<WrapperBase> && h) = default;

    Handle<WrapperBase> & operator=(Handle<WrapperBase> const& h) = default;
    Handle<WrapperBase> & operator=(Handle<WrapperBase> && h) = default;

    // (non-trivial) default destructor
    ~Handle() = default;

    // reimplement swap over HandleBase
    // DO NOT swap with a HandleBase or with a different Handle<T>
    void swap(Handle<WrapperBase> & other) {
      HandleBase::swap(static_cast<HandleBase &>(other));
      std::swap(type_, other.type_);
    }

    WrapperBase const* product() const {
      return static_cast<WrapperBase const*>(productStorage());
    }

    WrapperBase const* operator->() const {
      return product();
    }

    WrapperBase const& operator*() const {
      return *product();
    }

    std::type_info const& typeInfo() const {
      return *type_;
    }

  private:
    std::type_info const* type_ = nullptr;
  };


  // swap free function

  inline void swap(Handle<WrapperBase> & a, Handle<WrapperBase> & b)
  {
    a.swap(b);
  }


  // specialise the conversion from a BasicHandle into a Handle<WrapperBase>

  template <>
  inline
  void convert_handle(BasicHandle && bh, Handle<WrapperBase>& result)
  {
    if (bh.failedToGet()) {
      Handle<WrapperBase> h(std::move(bh.whyFailedFactory()));
      result = std::move(h);
      return;
    }
    WrapperBase const* wrapper = bh.wrapper();
    if (wrapper == nullptr) {
      handleimpl::throwInvalidReference();
    }
    if (not (wrapper->dynamicTypeInfo() == result.typeInfo())) {
      handleimpl::throwConvertTypeError(result.typeInfo(), bh.wrapper()->dynamicTypeInfo());
    }
    Handle<WrapperBase> h(wrapper, bh.provenance(), result.typeInfo());
    result = std::move(h);
  }


  // specialise OrphanHandle for a WrapperBase
    
  template <>
  class OrphanHandle<WrapperBase> : public OrphanHandleBase {
  public:
    // default constructed handles are invalid
    OrphanHandle() :
      OrphanHandleBase(),
      type_(nullptr)
    { }

    // does not take ownership of the WrapperBase
    // throws an exception if `type` is an invalid or unknown C++ class type
    OrphanHandle(WrapperBase const* wrapper, std::string const& type, ProductID const& id) :
      OrphanHandleBase(wrapper, id),
      type_(& TypeWithDict::byName(type).typeInfo())
    { }

    // does not take ownership of the WrapperBase
    // assumes `type` to be a valid and known C++ type 
    OrphanHandle(WrapperBase const* wrapper, std::type_info const& type, ProductID const& id) :
      OrphanHandleBase(wrapper, id),
      type_(& type)
    { }

    // copiable and moveable
    OrphanHandle(OrphanHandle<WrapperBase> const& h) = default;
    OrphanHandle(OrphanHandle<WrapperBase> && h) = default;

    OrphanHandle<WrapperBase> & operator=(OrphanHandle<WrapperBase> const& h) = default;
    OrphanHandle<WrapperBase> & operator=(OrphanHandle<WrapperBase> && h) = default;

    // default destructor
    ~OrphanHandle() = default;

    // reimplement swap over OrphanHandleBase
    // DO NOT swap with a OrphanHandleBase or with a different OrphanHandle<T>
    void swap(OrphanHandle<WrapperBase> & other) {
      OrphanHandleBase::swap(static_cast<OrphanHandleBase &>(other));
      std::swap(type_, other.type_);
    }

    WrapperBase const* product() const {
      return static_cast<WrapperBase const*>(productStorage());
    }

    WrapperBase const* operator->() const {
      return product();
    }

    WrapperBase const& operator*() const {
      return *product();
    }

    std::type_info const& typeInfo() const {
      return *type_;
    }

  private:
    std::type_info const* type_ = nullptr;
  };


  // swap free function

  inline void swap(OrphanHandle<WrapperBase> & a, OrphanHandle<WrapperBase> & b) {
    a.swap(b);
  }

  // specialise the Event methods for getting a WrapperBase

  template <>
  inline
  bool Event::getByLabel(InputTag const& tag, Handle<WrapperBase>& result) const
  {
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(result.typeInfo()), tag, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template <>
  inline
  bool Event::getByLabel(std::string const& label, std::string const& productInstanceName, Handle<WrapperBase>& result) const
  {
    result.clear();
    BasicHandle bh = provRecorder_.getByLabel_(TypeID(result.typeInfo()), label, productInstanceName, emptyString_, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template <>
  inline
  bool Event::getByToken(EDGetToken token, Handle<WrapperBase>& result) const
  {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(result.typeInfo()), PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  template <>
  inline
  bool Event::getByToken(EDGetTokenT<WrapperBase> token, Handle<WrapperBase>& result) const
  {
    result.clear();
    BasicHandle bh = provRecorder_.getByToken_(TypeID(result.typeInfo()), PRODUCT_TYPE, token, moduleCallingContext_);
    convert_handle(std::move(bh), result);  // throws on conversion error
    if (result.failedToGet()) {
      return false;
    }
    addToGotBranchIDs(*result.provenance());
    return true;
  }

  // specialise the Event methods for putting a WrapperBase

  template <>
  inline
  OrphanHandle<WrapperBase> Event::putImpl(EDPutToken::value_type index, std::unique_ptr<WrapperBase> product)
  {
    // the underlying collection `post_insert` is not called, as it is assumed
    // it was done before creating the Wrapper
    assert(index < putProducts().size());
    //putProducts()[index] = std::move(product->wrapper());
    putProducts()[index] = std::move(product);
    WrapperBase const* prod = putProducts()[index].get();
    auto const& prodType = prod->dynamicTypeInfo();
    auto const& prodID   = provRecorder_.getProductID(index);
    return OrphanHandle<WrapperBase>(prod, prodType, prodID);
  }

  template <>
  inline
  OrphanHandle<WrapperBase> Event::put(EDPutToken token, std::unique_ptr<WrapperBase> product)
  {
    auto const& typeInfo = product->dynamicTypeInfo();
    if (UNLIKELY(product.get() == nullptr)) {
      // null pointer is illegal
      principal_get_adapter_detail::throwOnPutOfNullProduct("Event", TypeID(typeInfo), provRecorder_.productInstanceLabel(token));
    }
    if (UNLIKELY(token.isUninitialized())) {
      principal_get_adapter_detail::throwOnPutOfUninitializedToken("Event", typeInfo);
    }
    if (UNLIKELY(provRecorder_.getTypeIDForPutTokenIndex(token.index()) != TypeID(typeInfo))) {
      principal_get_adapter_detail::throwOnPutOfWrongType(typeInfo, provRecorder_.getTypeIDForPutTokenIndex(token.index()));
    }
    return putImpl(token.index(), std::move(product));
  }

}

#endif // WrapperHandle_h
