#ifndef locking_ptr_h
#define locking_ptr_h

#include <memory>

#include "DQMServices/Core/interface/MonitorElement.h"

namespace detail {

  template <typename T>
  class locking_ptr_impl
  {
  public:
    locking_ptr_impl<T>(T * object, MonitorElement::LockType * lock) :
      object_(object),
      lock_(lock)
    {
      lock_->lock();
    }

    ~locking_ptr_impl<T>()
    {
      lock_->unlock();
    }

    T * operator->() {
      return object_;
    }

    const T * operator->() const {
      return object_;
    }

  protected:
    T * object_;
    MonitorElement::LockType * lock_;
  };

} // detail

template <typename T>
class locking_ptr
{
public:
  locking_ptr<T>() :
    object_(nullptr),
    lock_(nullptr)
  { }

  locking_ptr<T>(T * object, MonitorElement::LockType * lock) :
    object_(object),
    lock_(lock)
  { }

  locking_ptr<T>(locking_ptr<T> const& other) :
    object_(other.object_),
    lock_(other.lock_)
  { }

  template <typename U>
  locking_ptr<T>(locking_ptr<U> const& other) :
    object_(other.object_),
    lock_(other.lock_)
  { }

  ~locking_ptr<T>()
  { }

  detail::locking_ptr_impl<T> operator->() {
    return detail::locking_ptr_impl<T>(object_, lock_);
  }

  detail::locking_ptr_impl<const T> operator->() const {
    return detail::locking_ptr_impl<const T>(object_, lock_);
  }

  explicit operator bool() const noexcept {
    return (object_ != nullptr);
  }

  MonitorElement::LockType * mutex() const {
    return * lock_;
  }

protected:
  T * object_;
  MonitorElement::LockType * lock_;

private:
  template <typename U>
  friend class locking_ptr;
};


template <typename T>
locking_ptr<T> make_locking(T * object, MonitorElement::LockType * lock)
{
  return locking_ptr<T>(object, lock);
}

template <typename T>
const locking_ptr<const T> make_locking(const T * object, MonitorElement::LockType * lock)
{
  return locking_ptr<const T>(object, lock);
}

template <typename T>
locking_ptr<T> make_locking(T * object, MonitorElement::LockType & lock)
{
  return locking_ptr<T>(object, & lock);
}

template <typename T>
const locking_ptr<const T> make_locking(const T * object, MonitorElement::LockType & lock)
{
  return locking_ptr<const T>(object, & lock);
}


#endif // ! locking_ptr_h
