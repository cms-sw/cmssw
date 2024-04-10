#ifndef CondFormats_SerializationHelper_unique_void_ptr_h
#define CondFormats_SerializationHelper_unique_void_ptr_h
// -*- C++ -*-
//
// Package:     CondFormats/SerializationHelper
// Class  :     unique_void_ptr
//
/**\class unique_void_ptr unique_void_ptr.h "CondFormats/SerializationHelper/interface/unique_void_ptr.h"

 Description: Provides ownership of a const void*

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 31 May 2023 14:34:15 GMT
//

// system include files
#include <functional>

// user include files

// forward declarations

namespace cond::serialization {
  class unique_void_ptr {
  public:
    unique_void_ptr() noexcept : ptr_{nullptr} {}
    unique_void_ptr(const void* iPtr, std::function<void(const void*)> iDestructor) noexcept
        : ptr_(iPtr), destructor_(std::move(iDestructor)) {}

    ~unique_void_ptr() noexcept {
      if (destructor_) {
        destructor_(ptr_);
      }
    }

    unique_void_ptr(const unique_void_ptr&) = delete;             // stop default
    unique_void_ptr& operator=(const unique_void_ptr&) = delete;  // stop default

    unique_void_ptr(unique_void_ptr&& iOld) noexcept : ptr_(iOld.ptr_), destructor_(std::move(iOld.destructor_)) {
      iOld.ptr_ = nullptr;
    }
    unique_void_ptr& operator=(unique_void_ptr&& iOld) noexcept {
      unique_void_ptr tmp(std::move(*this));
      std::swap(ptr_, iOld.ptr_);
      std::swap(destructor_, iOld.destructor_);
      return *this;
    }
    // ---------- const member functions ---------------------
    const void* get() const noexcept { return ptr_; }

    // ---------- member functions --------------------------------
    const void* release() noexcept {
      auto tmp = ptr_;
      ptr_ = nullptr;
      return tmp;
    }

  private:
    // ---------- member data --------------------------------
    const void* ptr_;
    std::function<void(const void*)> destructor_;
  };
}  // namespace cond::serialization
#endif
