#ifndef CUDADataFormatsCommonHostProduct_H
#define CUDADataFormatsCommonHostProduct_H

#include <memory>

// a heterogeneous unique pointer...
template <typename T>
class HostProduct {
public:
  HostProduct() = default;  // make root happy
  ~HostProduct() = default;
  HostProduct(HostProduct&&) = default;
  HostProduct& operator=(HostProduct&&) = default;

  explicit HostProduct(std::unique_ptr<T>&& p) : std_ptr(std::move(p)) {}

  auto const* get() const { return std_ptr.get(); }

  auto const& operator*() const { return *get(); }

  auto const* operator->() const { return get(); }

private:
  std::unique_ptr<T> std_ptr;  //!
};

#endif
