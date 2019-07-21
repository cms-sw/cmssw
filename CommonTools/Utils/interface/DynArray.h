#ifndef CommonTools_Utils_DynArray_H
#define CommonTools_Utils_DynArray_H

template <typename T>
class DynArray {
public:
  T* a = nullptr;
  unsigned int s = 0;

public:
  using size_type = unsigned int;
  using value_type = T;
  using reference = T&;
  using const_reference = T const&;

  DynArray() {}

  explicit DynArray(unsigned char* storage) : a((T*)(storage)), s(0) {}

  DynArray(unsigned char* storage, unsigned int isize) : a((T*)(storage)), s(isize) {
    for (auto i = 0U; i < s; ++i)
      new ((begin() + i)) T();
  }
  DynArray(unsigned char* storage, unsigned int isize, T const& it) : a((T*)(storage)), s(isize) {
    for (auto i = 0U; i < s; ++i)
      new ((begin() + i)) T(it);
  }

  DynArray(DynArray const&) = delete;
  DynArray& operator=(DynArray const&) = delete;

  DynArray(DynArray&& other) {
    a = other.a;
    s = other.s;
    other.s = 0;
    other.a = nullptr;
  }
  DynArray& operator=(DynArray&& other) {
    a = other.a;
    s = other.s;
    other.s = 0;
    other.a = nullptr;
    return *this;
  }

  ~DynArray() {
    for (auto i = 0U; i < s; ++i)
      a[i].~T();
  }

  T& operator[](unsigned int i) { return a[i]; }
  T* begin() { return a; }
  T* end() { return a + s; }
  T& front() { return a[0]; }
  T& back() { return a[s - 1]; }

  T const& operator[](unsigned int i) const { return a[i]; }
  T const* begin() const { return a; }
  T const* end() const { return a + s; }
  unsigned int size() const { return s; }
  bool empty() const { return 0 == s; }

  T const* data() const { return a; }
  T const& front() const { return a[0]; }
  T const& back() const { return a[s - 1]; }

  void pop_back() {
    back().~T();
    --s;
  }
  void push_back(T const& t) {
    new ((begin() + s)) T(t);
    ++s;
  }
  void push_back(T&& t) {
    new ((begin() + s)) T(t);
    ++s;
  }
};

namespace dynarray {
  template <typename T>
  inline T num(T s) {
    return s > 0 ? s : T(1);
  }
}  // namespace dynarray

#define unInitDynArray(T, n, x)                                                \
  alignas(alignof(T)) unsigned char x##_storage[sizeof(T) * dynarray::num(n)]; \
  DynArray<T> x(x##_storage)
#define declareDynArray(T, n, x)                                               \
  alignas(alignof(T)) unsigned char x##_storage[sizeof(T) * dynarray::num(n)]; \
  DynArray<T> x(x##_storage, n)
#define initDynArray(T, n, x, i)                                               \
  alignas(alignof(T)) unsigned char x##_storage[sizeof(T) * dynarray::num(n)]; \
  DynArray<T> x(x##_storage, n, i)

#endif  // CommonTools_Utils_DynArray_H
