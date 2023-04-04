#ifndef AlpakaCore_AtomicPairCounter_h
#define AlpakaCore_AtomicPairCounter_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  class AtomicPairCounter {
  public:
    using c_type = unsigned long long int;

    ALPAKA_FN_HOST_ACC AtomicPairCounter() {}
    ALPAKA_FN_HOST_ACC AtomicPairCounter(c_type i) { counter.ac = i; }

    ALPAKA_FN_HOST_ACC AtomicPairCounter& operator=(c_type i) {
      counter.ac = i;
      return *this;
    }

    struct Counters {
      uint32_t n;  // in a "One to Many" association is the number of "One"
      uint32_t m;  // in a "One to Many" association is the total number of associations
    };

    union Atomic2 {
      Counters counters;
      c_type ac;
    };

    static constexpr c_type incr = 1UL << 32;

    ALPAKA_FN_ACC Counters get() const { return counter.counters; }

    // increment n by 1 and m by i.  return previous value
    template <typename TAcc>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE Counters add(const TAcc& acc, uint32_t i) {
      c_type c = i;
      c += incr;

      Atomic2 ret;
      ret.ac = alpaka::atomicAdd(acc, &counter.ac, c, alpaka::hierarchy::Blocks{});
      return ret.counters;
    }

  private:
    Atomic2 counter;
  };

}  // namespace cms::alpakatools

#endif  // AlpakaCore_AtomicPairCounter_h
