#pragma once
#include <memory>
#include <new>

#include <iostream>
class SimplePoolAllocator;

namespace memoryPool {

  enum Where { onCPU, onDevice, onHost, unified };

  class DeleterBase {
  public:
    explicit DeleterBase(SimplePoolAllocator* pool, void* stream) : m_pool(pool), m_stream(stream) {}
    virtual ~DeleterBase() = default;
    virtual void operator()(int bucket, uint64_t count) = 0;

    SimplePoolAllocator* pool() const { return m_pool; }
    void* stream() const { return m_stream; }

  protected:
    SimplePoolAllocator* m_pool;
    void* m_stream;
  };

  class Deleter {
  public:
    Deleter() = default;
    explicit Deleter(std::shared_ptr<DeleterBase> const& del) : me(del) {}
    explicit Deleter(std::shared_ptr<DeleterBase>&& del) : me(del) {}

    void set(std::shared_ptr<DeleterBase> const& del) { me = del; }
    std::shared_ptr<DeleterBase> const& get() const { return me; }

    void operator()(int bucket, uint64_t count) {
      if (!me) {
        std::cout << "deleter w/o implementation!!!" << std::endl;
        throw std::bad_alloc();
      }
      if (bucket < 0)
        std::cout << "delete with negative bucket!!!" << std::endl;
      (*me)(bucket, count);
    }

    SimplePoolAllocator* pool() const { return me->pool(); }
    void* stream() const { return me->stream(); }

  private:
    std::shared_ptr<DeleterBase> me;  //!
  };

  template <typename T>
  class Buffer {
  public:
    typedef T value_type;
    typedef T* pointer;
    typedef T& reference;
    typedef T const* const_pointer;
    typedef T const& const_reference;

    using Tuple = std::tuple<T*, int, uint64_t>;

    Buffer() = default;
    Buffer(T* p, int bucket, uint64_t count) : m_p(p), m_bucket(bucket), m_count(count) {}
    Buffer(T* p, int bucket, uint64_t count, Deleter const& del)
        : m_deleter(del), m_p(p), m_bucket(bucket), m_count(count) {}
    Buffer(T* p, int bucket, uint64_t count, Deleter&& del)
        : m_deleter(del), m_p(p), m_bucket(bucket), m_count(count) {}
    Buffer(Tuple const& rh, Deleter const& del)
        : m_deleter(del), m_p(std::get<0>(rh)), m_bucket(std::get<1>(rh)), m_count(std::get<2>(rh)) {}
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;

    template <typename U>
    Buffer(Buffer<U>&& rh) : Buffer(rh.release(), rh.deleter()) {}
    template <typename U>
    Buffer& operator=(Buffer<U>&& rh) {
      reset(rh.release());
      m_deleter = rh.deleter();
      return *this;
    }

    ~Buffer() {
      // assert(m_p == pool()->pointer(m_bucket));
      if (m_p)
        m_deleter(m_bucket, m_count);
    }

    pointer get() { return m_p; }
    const_pointer get() const { return m_p; }
    reference operator*() { return *m_p; }
    const_reference operator*() const { return *m_p; }
    pointer operator->() { return get(); }
    const_pointer operator->() const { return get(); }
    reference operator[](int i) { return m_p[i]; }
    const_reference operator[](int i) const { return m_p[i]; }

    Deleter& deleter() { return m_deleter; }
    Deleter const& deleter() const { return m_deleter; }
    SimplePoolAllocator* pool() const { return deleter().pool(); }

    int bucket() const { return m_bucket; }
    uint64_t count() const { return m_count; }

    Tuple release() {
      auto ret = std::make_tuple(m_p, m_bucket, m_count);
      m_p = nullptr;
      m_bucket = -1;
      return ret;
    }
    void reset() {
      if (m_p)
        m_deleter(m_bucket, m_count);
      m_p = nullptr;
      m_bucket = -1;
    }
    void reset(Tuple const& rh) {
      if (m_p)
        m_deleter(m_bucket, m_count);
      m_p = std::get<0>(rh);
      m_bucket = std::get<1>(rh);
      m_count = std::get<2>(rh);
    }

  private:
    Deleter m_deleter;      //!
    pointer m_p = nullptr;  //!
    int m_bucket = -1;      //!
    uint64_t m_count;       //!
  };

}  // namespace memoryPool
