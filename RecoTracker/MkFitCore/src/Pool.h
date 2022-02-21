#ifndef RecoTracker_MkFitCore_src_Pool_h
#define RecoTracker_MkFitCore_src_Pool_h

#include "oneapi/tbb/concurrent_queue.h"

namespace mkfit {

  /**
   * Pool for helper objects. All functions are thread safe.
   */
  template <typename TT>
  class Pool {
  public:
    Pool() = default;

    ~Pool() {
      TT *x = nullptr;
      while (m_stack.try_pop(x)) {
        destroy(x);
      }
    }

    size_t size() const { return m_stack.unsafe_size(); }

    void populate(int threads = Config::numThreadsFinder) {
      for (int i = 0; i < threads; ++i) {
        m_stack.push(create());
      }
    }

    auto makeOrGet() {
      TT *x = nullptr;
      if (not m_stack.try_pop(x)) {
        x = create();
      }
      auto deleter = [this](TT *ptr) { this->addBack(ptr); };
      return std::unique_ptr<TT, decltype(deleter)>(x, std::move(deleter));
    }

  private:
    TT *create() { return new (std::aligned_alloc(64, sizeof(TT))) TT; };

    void destroy(TT *x) {
      x->~TT();
      std::free(x);
    };

    void addBack(TT *x) { m_stack.push(x); }

    tbb::concurrent_queue<TT *> m_stack;
  };

}  // end namespace mkfit
#endif
