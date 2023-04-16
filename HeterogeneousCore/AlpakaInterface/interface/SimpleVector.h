#ifndef HeterogeneousCore_AlpakaInterface_interface_SimpleVector_h
#define HeterogeneousCore_AlpakaInterface_interface_SimpleVector_h

//  author: Felice Pantaleo, CERN, 2018
//  alpaka integration in cmssw: Adriano Di Florio, 2022

#include <type_traits>
#include <utility>
#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  template <class T>
  struct SimpleVector {
    constexpr SimpleVector() = default;

    // ownership of m_data stays within the caller
    constexpr void construct(int capacity, T *data) {
      m_size = 0;
      m_capacity = capacity;
      m_data = data;
    }

    inline constexpr int push_back_unsafe(const T &element) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < m_capacity) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    template <class... Ts>
    constexpr int emplace_back_unsafe(Ts &&...args) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < m_capacity) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    ALPAKA_FN_ACC inline T &back() { return m_data[m_size - 1]; }

    ALPAKA_FN_ACC inline const T &back() const {
      if (m_size > 0) {
        return m_data[m_size - 1];
      } else
        return T();  //undefined behaviour
    }

    // thread-safe version of the vector, when used in a CUDA kernel
    template <typename TAcc>
    ALPAKA_FN_ACC int push_back(const TAcc &acc, const T &element) {
      auto previousSize = alpaka::atomicAdd(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
      if (previousSize < m_capacity) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        alpaka::atomicSub(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    template <typename TAcc, class... Ts>
    ALPAKA_FN_ACC int emplace_back(const TAcc &acc, Ts &&...args) {
      auto previousSize = alpaka::atomicAdd(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
      if (previousSize < m_capacity) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        alpaka::atomicSub(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    // thread safe version of resize
    template <typename TAcc>
    ALPAKA_FN_ACC int extend(const TAcc &acc, int size = 1) {
      auto previousSize = alpaka::atomicAdd(acc, &m_size, size, alpaka::hierarchy::Blocks{});
      if (previousSize < m_capacity) {
        return previousSize;
      } else {
        alpaka::atomicSub(acc, &m_size, size, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    template <typename TAcc>
    ALPAKA_FN_ACC int shrink(const TAcc &acc, int size = 1) {
      auto previousSize = alpaka::atomicSub(acc, &m_size, size, alpaka::hierarchy::Blocks{});
      if (previousSize >= size) {
        return previousSize - size;
      } else {
        alpaka::atomicAdd(acc, &m_size, size, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    inline constexpr bool empty() const { return m_size <= 0; }
    inline constexpr bool full() const { return m_size >= m_capacity; }
    inline constexpr T &operator[](int i) { return m_data[i]; }
    inline constexpr const T &operator[](int i) const { return m_data[i]; }
    inline constexpr void reset() { m_size = 0; }
    inline constexpr int size() const { return m_size; }
    inline constexpr int capacity() const { return m_capacity; }
    inline constexpr T const *data() const { return m_data; }
    inline constexpr void resize(int size) { m_size = size; }
    inline constexpr void set_data(T *data) { m_data = data; }

  private:
    int m_size;
    int m_capacity;

    T *m_data;
  };

  // ownership of m_data stays within the caller
  template <class T>
  SimpleVector<T> make_SimpleVector(int capacity, T *data) {
    SimpleVector<T> ret;
    ret.construct(capacity, data);
    return ret;
  }

  // ownership of m_data stays within the caller
  template <class T>
  SimpleVector<T> *make_SimpleVector(SimpleVector<T> *mem, int capacity, T *data) {
    auto ret = new (mem) SimpleVector<T>();
    ret->construct(capacity, data);
    return ret;
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_SimpleVector_h
