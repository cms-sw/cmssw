#ifndef RecoTracker_MkFitCore_src_Pool_h
#define RecoTracker_MkFitCore_src_Pool_h

#include "Matriplex/Memory.h"
#include "oneapi/tbb/enumerable_thread_specific.h"
#include <memory>
#include <cassert>

namespace mkfit {

  /**
   * Pool for helper objects.  Helper objects are allocated per-thread, and are
   * assumed that use does not cross thread boundaries and that helper objects
   * are not used recursively.  There are assertions to enforce these assummptions.
   */
  template <typename TT>
  class Pool {
  public:
    Pool() : m_pool(&finit) {}
    ~Pool() = default;

    void clear() { m_pool.clear(); }
    size_t size() const { return m_pool.size(); }

    // for thread-specific pooling best to allocate in-thread
    void populate(int threads = Config::numThreadsFinder) {}

    auto makeOrGet() {
      bool exists;
      auto &m = m_pool.local(exists);
      TT *x;
      if (exists) {
        x = m.release();
        assert(x);
      } else {
        x = create();
      }
      auto recycler = [this](TT *ptr) { this->recycle(ptr); };
      return std::unique_ptr<TT, decltype(recycler)>(x, recycler);
    }

  private:
    TT *create() { return new (Matriplex::aligned_alloc64(sizeof(TT))) TT; };

    static void destroy(TT *x) {
      x->~TT();
      std::free(x);
    };
    using poolptr = std::unique_ptr<TT, decltype(&destroy)>;

    static auto finit() { return poolptr(nullptr, &destroy); }

    void recycle(TT *x) {
      assert(m_pool.local().get() == nullptr);
      m_pool.local().reset(x);
    }

    tbb::enumerable_thread_specific<poolptr> m_pool;
  };

}  // end namespace mkfit
#endif
