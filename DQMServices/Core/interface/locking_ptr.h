#ifndef locking_ptr_h
#define locking_ptr_h

#include <memory>

namespace detail {

  class lock_base
  {
  public:
    lock_base() = default;
    virtual ~lock_base() = default;

    virtual void lock() const = 0;
    virtual void unlock() const = 0;
  };

  template <typename L>
  class lock_impl : public lock_base
  {
  public:
    lock_impl<L>(L & lock) :
      lock_(lock)
    { }

    void lock() const final
    {
      lock_.lock();
    }

    void unlock() const final
    {
      lock_.unlock();
    }

  private:
    L & lock_;
  };


  template <typename T>
  class locking_ptr_impl
  {
  public:
    locking_ptr_impl<T>(T * object, lock_base * lock) :
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
    lock_base * lock_;
  };

} // detail

template <typename T>
class locking_ptr
{
public:
  locking_ptr<T>() :
    object_(nullptr),
    lock_()
  { }

  template <typename L>
  locking_ptr<T>(T * object, L & lock) :
    object_(object),
    lock_(std::make_shared<detail::lock_impl<L>>(lock))
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
    return detail::locking_ptr_impl<T>(object_, lock_.get());
  }

  detail::locking_ptr_impl<const T> operator->() const {
    return detail::locking_ptr_impl<const T>(object_, lock_.get());
  }

  explicit operator bool() const noexcept {
    return (object_ != nullptr);
  }

protected:
  T * object_;
  std::shared_ptr<detail::lock_base> lock_;

private:
  template <typename U>
  friend class locking_ptr;
};


template <typename T, typename L>
locking_ptr<T> make_locking(T * object, L & lock)
{
  return locking_ptr<T>(object, lock);
}

template <typename T, typename L>
const locking_ptr<const T> make_locking(const T * object, L & lock)
{
  return locking_ptr<const T>(object, lock);
}


#endif // ! locking_ptr_h
