#ifndef DataFormats_Common_Wrapper_h
#define DataFormats_Common_Wrapper_h

/*----------------------------------------------------------------------

Wrapper: A template wrapper around EDProducts to hold the product ID.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/TrivialCopyTraits.h"
#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/WrapperDetail.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <span>
#include <string>
#include <typeinfo>
#include <vector>

namespace edm {
  template <typename T>
  class Wrapper : public WrapperBase {
  public:
    typedef T value_type;
    typedef T wrapped_type;  // used with the dictionary to identify Wrappers
    Wrapper() : WrapperBase(), obj{construct_()}, present(false) {}
    explicit Wrapper(std::unique_ptr<T> ptr);
    Wrapper(Wrapper<T> const& rh) = delete;             // disallow copy construction
    Wrapper<T>& operator=(Wrapper<T> const&) = delete;  // disallow assignment

    template <typename... Args>
    explicit Wrapper(Emplace, Args&&...);
    ~Wrapper() override {}
    T const* product() const { return (present ? &obj : nullptr); }
    T const* operator->() const { return product(); }

    T& bareProduct() { return obj; }

    //these are used by FWLite
    static std::type_info const& productTypeInfo() { return typeid(T); }
    static std::type_info const& typeInfo() { return typeid(Wrapper<T>); }

    // the constructor takes ownership of T*
    Wrapper(T*);

    //Used by ROOT storage
    CMS_CLASS_VERSION(4)

  private:
    constexpr T construct_() {
      if constexpr (requires { T(); }) {
        return T();
      } else {
        return T(edm::kUninitialized);
      }
    }

    bool isPresent_() const override { return present; }
    void markAsPresent_() override { present = true; }
    std::type_info const& dynamicTypeInfo_() const override { return typeid(T); }
    std::type_info const& wrappedTypeInfo_() const override { return typeid(Wrapper<T>); }

    std::type_info const& valueTypeInfo_() const override;
    std::type_info const& memberTypeInfo_() const override;
    bool isMergeable_() const override;
    bool mergeProduct_(WrapperBase const* newProduct) override;
    bool hasIsProductEqual_() const override;
    bool isProductEqual_(WrapperBase const* newProduct) const override;
    bool hasSwap_() const override;
    void swapProduct_(WrapperBase* newProduct) override;

    void do_fillView(ProductID const& id,
                     std::vector<void const*>& pointers,
                     FillViewHelperVector& helpers) const override;
    void do_setPtr(std::type_info const& iToType, unsigned long iIndex, void const*& oPtr) const override;
    void do_fillPtrVector(std::type_info const& iToType,
                          std::vector<unsigned long> const& iIndices,
                          std::vector<void const*>& oPtr) const override;

    std::shared_ptr<soa::TableExaminerBase> tableExaminer_() const override;

    bool hasTrivialCopyTraits_() const override;
    bool hasTrivialCopyProperties_() const override;
    void trivialCopyInitialize_(edm::AnyBuffer const& args) override;
    edm::AnyBuffer trivialCopyParameters_() const override;
    std::vector<std::span<const std::byte>> trivialCopyRegions_() const override;
    std::vector<std::span<std::byte>> trivialCopyRegions_() override;
    void trivialCopyFinalize_() override;

  private:
    T obj;
    bool present;
  };

  template <typename T>
  Wrapper<T>::Wrapper(std::unique_ptr<T> ptr) : WrapperBase(), obj{construct_()}, present(ptr.get() != nullptr) {
    if (present) {
      obj = std::move(*ptr);
    }
  }

  template <typename T>
  template <typename... Args>
  Wrapper<T>::Wrapper(Emplace, Args&&... args) : WrapperBase(), obj(std::forward<Args>(args)...), present(true) {}

  template <typename T>
  Wrapper<T>::Wrapper(T* ptr) : WrapperBase(), present(ptr != 0), obj{construct_()} {
    std::unique_ptr<T> temp(ptr);
    if (present) {
      obj = std::move(*ptr);
    }
  }

  template <typename T>
  inline std::type_info const& Wrapper<T>::valueTypeInfo_() const {
    return detail::getValueType<T>()();
  }

  template <typename T>
  inline std::type_info const& Wrapper<T>::memberTypeInfo_() const {
    return detail::getMemberType<T>()();
  }

  template <typename T>
  inline bool Wrapper<T>::isMergeable_() const {
    if constexpr (requires(T& a, T const& b) { a.mergeProduct(b); }) {
      return true;
    }
    return false;
  }

  template <typename T>
  inline bool Wrapper<T>::mergeProduct_(WrapperBase const* newProduct) {
    Wrapper<T> const* wrappedNewProduct = dynamic_cast<Wrapper<T> const*>(newProduct);
    assert(wrappedNewProduct != nullptr);
    if constexpr (requires(T& a, T const& b) { a.mergeProduct(b); }) {
      return obj.mergeProduct(wrappedNewProduct->obj);
    }
    return true;
  }

  template <typename T>
  inline bool Wrapper<T>::hasIsProductEqual_() const {
    if constexpr (requires(T& a, T const& b) { a.isProductEqual(b); }) {
      return true;
    }
    return false;
  }

  template <typename T>
  inline bool Wrapper<T>::isProductEqual_(WrapperBase const* newProduct) const {
    Wrapper<T> const* wrappedNewProduct = dynamic_cast<Wrapper<T> const*>(newProduct);
    assert(wrappedNewProduct != nullptr);
    if constexpr (requires(T& a, T const& b) { a.isProductEqual(b); }) {
      return obj.isProductEqual(wrappedNewProduct->obj);
    }
    return true;
  }

  template <typename T>
  inline bool Wrapper<T>::hasSwap_() const {
    if constexpr (requires(T& a, T& b) { a.swap(b); }) {
      return true;
    }
    return false;
  }

  template <typename T>
  inline void Wrapper<T>::swapProduct_(WrapperBase* newProduct) {
    Wrapper<T>* wrappedNewProduct = dynamic_cast<Wrapper<T>*>(newProduct);
    assert(wrappedNewProduct != nullptr);
    if constexpr (requires(T& a, T& b) { a.swap(b); }) {
      obj.swap(wrappedNewProduct->obj);
    }
  }

  namespace soa {
    template <class T>
    struct MakeTableExaminer {
      static std::shared_ptr<edm::soa::TableExaminerBase> make(void const*) {
        return std::shared_ptr<edm::soa::TableExaminerBase>{};
      }
    };
  }  // namespace soa

  template <typename T>
  inline std::shared_ptr<edm::soa::TableExaminerBase> Wrapper<T>::tableExaminer_() const {
    return soa::MakeTableExaminer<T>::make(&obj);
  }

  template <typename T>
  inline bool Wrapper<T>::hasTrivialCopyTraits_() const {
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return true;
    }
    return false;
  }

  template <typename T>
  inline bool Wrapper<T>::hasTrivialCopyProperties_() const {
    if constexpr (requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      return true;
    }
    return false;
  }

  template <typename T>
  void Wrapper<T>::trivialCopyInitialize_([[maybe_unused]] edm::AnyBuffer const& args) {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    // if edm::TrivialCopyTraits<T>::Properties is not defined, do not call initialize()
    if constexpr (not requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      return;
    } else
      // if edm::TrivialCopyTraits<T>::Properties is void, call initialize() without any additional arguments
      if constexpr (std::is_same_v<typename edm::TrivialCopyTraits<T>::Properties, void>) {
        edm::TrivialCopyTraits<T>::initialize(obj);
      } else
      // if edm::TrivialCopyTraits<T>::Properties is not void, cast args to Properties and pass it as an additional argument to initialize()
      {
        edm::TrivialCopyTraits<T>::initialize(obj, args.cast_to<typename edm::TrivialCopyTraits<T>::Properties>());
      }
  }

  template <typename T>
  inline edm::AnyBuffer Wrapper<T>::trivialCopyParameters_() const {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    // if edm::TrivialCopyTraits<T>::Properties is not defined, do not call properties()
    if constexpr (not requires { typename edm::TrivialCopyTraits<T>::Properties; }) {
      return {};
    } else
      // if edm::TrivialCopyTraits<T>::Properties is void, do not call properties()
      if constexpr (std::is_same_v<typename edm::TrivialCopyTraits<T>::Properties, void>) {
        return {};
      } else
      // if edm::TrivialCopyTraits<T>::Properties is not void, call properties() and wrap the result in an edm::AnyBuffer
      {
        typename edm::TrivialCopyTraits<T>::Properties p = edm::TrivialCopyTraits<T>::properties(obj);
        return edm::AnyBuffer(p);
      }
  }

  template <typename T>
  inline std::vector<std::span<const std::byte>> Wrapper<T>::trivialCopyRegions_() const {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    if constexpr (requires(T const& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return edm::TrivialCopyTraits<T>::regions(obj);
    } else {
      throw edm::Exception(edm::errors::LogicError)
          << "edm::TrivialCopyTraits<T>::regions(const T&) is not defined for type "
          << edm::typeDemangle(typeid(T).name());
      return {};
    }
  }

  template <typename T>
  inline std::vector<std::span<std::byte>> Wrapper<T>::trivialCopyRegions_() {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::regions(t); }) {
      return edm::TrivialCopyTraits<T>::regions(obj);
    } else {
      throw edm::Exception(edm::errors::LogicError)
          << "edm::TrivialCopyTraits<T>::regions(const T&) is not defined for type "
          << edm::typeDemangle(typeid(T).name());
      return {};
    }
  }

  template <typename T>
  inline void Wrapper<T>::trivialCopyFinalize_() {
    if (not present) {
      throw edm::Exception(edm::errors::LogicError) << "Attempt to access an empty Wrapper";
    }
    if constexpr (requires(T& t) { edm::TrivialCopyTraits<T>::finalize(t); }) {
      edm::TrivialCopyTraits<T>::finalize(obj);
    }
  }

}  // namespace edm

#include "DataFormats/Common/interface/WrapperView.icc"

#endif  // DataFormats_Common_Wrapper_h
