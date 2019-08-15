#ifndef GENERS_IOISSHAREDPTR_HH_
#define GENERS_IOISSHAREDPTR_HH_

#include <memory>

namespace gs {
  template <class T>
  struct IOIsSharedPtr {
    enum { value = 0 };
  };

  template <class T>
  struct IOIsSharedPtr<std::shared_ptr<T>> {
    enum { value = 1 };
  };

  template <class T>
  struct IOIsSharedPtr<const std::shared_ptr<T>> {
    enum { value = 1 };
  };

  template <class T>
  struct IOIsSharedPtr<volatile std::shared_ptr<T>> {
    enum { value = 1 };
  };

  template <class T>
  struct IOIsSharedPtr<const volatile std::shared_ptr<T>> {
    enum { value = 1 };
  };
}  // namespace gs

#endif  // GENERS_IOISSHAREDPTR_HH_
