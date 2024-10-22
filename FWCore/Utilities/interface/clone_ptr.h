#ifndef FWCore_Utilities_clone_ptr_h
#define FWCore_Utilities_clone_ptr_h

#include <memory>

namespace extstd {

  /* modify unique_ptr behaviour adding a "cloning" copy-constructor and assignment-operator
   */
  template <typename T>
  struct clone_ptr {
    clone_ptr() noexcept : ptr_() {}
    template <typename... Args>
    explicit clone_ptr(Args&&... args) noexcept : ptr_(std::forward<Args>(args)...) {}

    clone_ptr(clone_ptr const& rh) : ptr_(rh ? rh->clone() : nullptr) {}
    clone_ptr(clone_ptr&& rh) noexcept : ptr_(std::move(rh.ptr_)) {}

    clone_ptr& operator=(clone_ptr const& rh) {
      if (&rh != this)
        this->reset(rh ? rh->clone() : nullptr);
      return *this;
    }
    clone_ptr& operator=(clone_ptr&& rh) noexcept {
      if (&rh != this)
        ptr_ = std::move(rh.ptr_);
      return *this;
    }

    operator bool() const { return static_cast<bool>(ptr_); }

    template <typename U>
    clone_ptr(clone_ptr<U> const& rh) : ptr_(rh ? rh->clone() : nullptr) {}
    template <typename U>
    clone_ptr(clone_ptr<U>&& rh) noexcept : ptr_(std::move(rh.ptr_)) {}

    template <typename U>
    clone_ptr& operator=(clone_ptr<U> const& rh) {
      if (&rh != this)
        this->reset(rh ? rh->clone() : nullptr);
      return *this;
    }
    template <typename U>
    clone_ptr& operator=(clone_ptr<U>&& rh) noexcept {
      if (&rh != this)
        ptr_ = std::move(rh.ptr_);
      return *this;
    }

    T* operator->() const { return ptr_.get(); }

    T& operator*() const { return *ptr_; }

    T* get() const { return ptr_.get(); }

    void reset(T* iValue) { ptr_.reset(iValue); }

  private:
    std::unique_ptr<T> ptr_;
  };

}  // namespace extstd

#endif
