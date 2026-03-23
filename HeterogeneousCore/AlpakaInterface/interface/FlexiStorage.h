
#ifndef HeterogeneousCore_AlpakaInterface_interface_FlexiStorage_h
#define HeterogeneousCore_AlpakaInterface_interface_FlexiStorage_h

#include <cstdint>
#include <limits>

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {
  class FlexiStorageBase {
  public:
    using size_type = uint32_t;
  };

  inline constexpr FlexiStorageBase::size_type kDynamicSize = std::numeric_limits<FlexiStorageBase::size_type>::max();
  inline constexpr FlexiStorageBase::size_type kOverflow = std::numeric_limits<FlexiStorageBase::size_type>::max() - 1;

  template <typename I, FlexiStorageBase::size_type S>
  class FlexiStorage : public FlexiStorageBase {
    static_assert(S != kOverflow, "FlexiStorage<I, S>: S must not be kOverflow");

  public:
    constexpr size_type capacity() const { return S; }

    constexpr I& operator[](size_type i) { return m_v[i]; }
    constexpr const I& operator[](size_type i) const { return m_v[i]; }

    constexpr I* data() { return m_v; }
    constexpr I const* data() const { return m_v; }

  private:
    I m_v[S];
  };

  template <typename I>
  class FlexiStorage<I, kDynamicSize> : public FlexiStorageBase {
  public:
    constexpr void init(I* v, size_type s) {
      ALPAKA_ASSERT_ACC((s != kDynamicSize) && "FlexiStorage::init(v, s): s must not be kDynamicSize");
      ALPAKA_ASSERT_ACC((s != kOverflow) && "FlexiStorage::init(v, s): s must not be kOverflow");
      m_v = v;
      m_capacity = s;
    }

    constexpr size_type capacity() const { return m_capacity; }

    constexpr I& operator[](size_type i) { return m_v[i]; }
    constexpr const I& operator[](size_type i) const { return m_v[i]; }

    constexpr I* data() { return m_v; }
    constexpr I const* data() const { return m_v; }

  private:
    I* m_v;
    size_type m_capacity;
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_FlexiStorage_h
