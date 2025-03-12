#ifndef DataFormats_Common_interface_AnyBuffer_h
#define DataFormats_Common_interface_AnyBuffer_h

#include <type_traits>
#include <typeinfo>

#include <boost/container/small_vector.hpp>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"

namespace edm {

  class AnyBuffer {
  public:
    AnyBuffer() = default;

    template <typename T>
    AnyBuffer(T const& t)
      requires(std::is_trivially_copyable_v<T>)
        : storage_(reinterpret_cast<std::byte const*>(&t), reinterpret_cast<std::byte const*>(&t) + sizeof(T)),
          typeid_(&typeid(std::remove_cv_t<T>)) {}

    template <typename T>
    T& cast_to()
      requires(std::is_trivially_copyable_v<T>)
    {
      if (empty()) {
        throw edm::Exception(edm::errors::LogicError)
            << "Attempt to read an object of type " << edm::typeDemangle(typeid(T).name())
            << " from an empty AnyBuffer";
      }
      if (typeid(std::remove_cv_t<T>) != *typeid_) {
        throw edm::Exception(edm::errors::LogicError)
            << "Attempt to read an object of type " << edm::typeDemangle(typeid(T).name())
            << " from an AnyBuffer holding an object of type " << edm::typeDemangle(typeid_->name());
      }
      return *reinterpret_cast<T*>(storage_.data());
    }

    template <typename T>
    T const& cast_to() const
      requires(std::is_trivially_copyable_v<T>)
    {
      if (empty()) {
        throw edm::Exception(edm::errors::LogicError)
            << "Attempt to read an object of type " << edm::typeDemangle(typeid(T).name())
            << " from an empty AnyBuffer";
      }
      if (typeid(std::remove_cv_t<T>) != *typeid_) {
        throw edm::Exception(edm::errors::LogicError)
            << "Attempt to read an object of type " << edm::typeDemangle(typeid(T).name())
            << " from an AnyBuffer holding an object of type " << edm::typeDemangle(typeid_->name());
      }
      return *reinterpret_cast<T const*>(storage_.data());
    }

    bool empty() const { return typeid_ == nullptr; }

    std::byte* data() { return storage_.data(); }

    std::byte const* data() const { return storage_.data(); }

    size_t size_bytes() const { return storage_.size(); }

  private:
    boost::container::small_vector<std::byte, 32> storage_;  // arbitrary small vector size to fit AnyBuffer in 64 bytes
    std::type_info const* typeid_ = nullptr;
  };

}  // namespace edm

#endif  // DataFormats_Common_interface_AnyBuffer_h
