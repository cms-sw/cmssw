#ifndef FWCore_Framework_interface_WrapperBaseHandle_h
#define FWCore_Framework_interface_WrapperBaseHandle_h
/*
 Description: Allows interaction with data in the Event without actually using the C++ class

 Usage:
    The Handle<WrapperBase> allows one to get data back from the edm::Event as an edm::Wrappter<T>
  via a polymorphic pointer of type edm::WrapperBase, instead of as the actual C++ class type.

  // make a handle to hold an instance of MyClass
  edm::Handle<edm::WrapperBase> handle(typeid(MyClass));
  event.getByToken(token, handle);

  // handle.product() returns a polymorphic pointer of type edm::WrapperBase to the underlying
  // edm::Wrapper<MyClass>
  assert(handle.product()->dynamicTypeInfo() == typeid(MyClass));
  edm::Wrapper<MyClass> const* wrapper = dynamic_cast<edm::Wrapper<MyClass> const*>(handle.product());
*/

// c++ include files
#include <memory>
#include <string>

// CMSSW include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"

// forward declarations
namespace edm {

  class Provenance;

  template <>
  class Handle<WrapperBase> {
  public:
    explicit Handle(std::type_info const& type) : type_(type), product_(nullptr), prov_(nullptr) {}

    explicit Handle(TypeID type) : type_(type), product_(nullptr), prov_(nullptr) {
      if (not type_) {
        throw Exception(errors::NotFound, "Handle<WrapperBase> given an invalid type.");
      }
    }

    Handle(WrapperBase const* product, Provenance const* prov)
        : type_(product->dynamicTypeInfo()), product_(product), prov_(prov) {
      assert(product_);
      assert(prov_);
    }

    // Reimplement the interface of HandleBase

    void clear() {
      product_ = nullptr;
      prov_ = nullptr;
      whyFailedFactory_ = nullptr;
    }

    bool isValid() const { return nullptr != product_ and nullptr != prov_; }

    bool failedToGet() const { return bool(whyFailedFactory_); }

    Provenance const* provenance() const { return prov_; }

    ProductID id() const { return prov_->productID(); }

    std::shared_ptr<cms::Exception> whyFailed() const {
      if (whyFailedFactory_.get()) {
        return whyFailedFactory_->make();
      }
      return std::shared_ptr<cms::Exception>();
    }

    std::shared_ptr<HandleExceptionFactory const> const& whyFailedFactory() const { return whyFailedFactory_; }

    explicit operator bool() const { return isValid(); }

    bool operator!() const { return not isValid(); }

    // Reimplement the interface of Handle<T>

    WrapperBase const* product() const {
      if (this->failedToGet()) {
        whyFailedFactory_->make()->raise();
      }
      return product_;
    }

    WrapperBase const* operator->() const { return this->product(); }
    WrapperBase const& operator*() const { return *(this->product()); }

    // Additional methods

    TypeID const& type() const { return type_; }

    void setWhyFailedFactory(std::shared_ptr<HandleExceptionFactory const> const& iWhyFailed) {
      whyFailedFactory_ = iWhyFailed;
    }

  private:
    TypeID type_;
    WrapperBase const* product_;
    Provenance const* prov_;
    std::shared_ptr<HandleExceptionFactory const> whyFailedFactory_;
  };

  // Specialize convert_handle for Handle<WrapperBase>
  void convert_handle(BasicHandle&& orig, Handle<WrapperBase>& result);

  // Specialize the Event's getByToken method to work with a Handle<WrapperBase>
  template <>
  bool Event::getByToken<WrapperBase>(EDGetToken token, Handle<WrapperBase>& result) const;

}  // namespace edm

#endif  // FWCore_Framework_interface_WrapperBaseHandle_h
