#ifndef HeterogeneousCore_CUDAUtilities_interface_FlexiStorage_h
#define HeterogeneousCore_CUDAUtilities_interface_FlexiStorage_h

#include <cstdint>

namespace cms {
  namespace cuda {

    template <typename I, int S>
    class FlexiStorage {
    public:
      constexpr int capacity() const { return S; }

      constexpr I& operator[](int i) { return m_v[i]; }
      constexpr const I& operator[](int i) const { return m_v[i]; }

      constexpr I* data() { return m_v; }
      constexpr I const* data() const { return m_v; }

    private:
      I m_v[S];
    };

    template <typename I>
    class FlexiStorage<I, -1> {
    public:
      constexpr void init(I* v, int s) {
        m_v = v;
        m_capacity = s;
      }

      constexpr int capacity() const { return m_capacity; }

      constexpr I& operator[](int i) { return m_v[i]; }
      constexpr const I& operator[](int i) const { return m_v[i]; }

      constexpr I* data() { return m_v; }
      constexpr I const* data() const { return m_v; }

    private:
      I* m_v;
      int m_capacity;
    };

  }  // namespace cuda

}  // namespace cms

#endif
