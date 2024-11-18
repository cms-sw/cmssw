#include <type_traits>
#include <utility>
#include <alpaka/alpaka.hpp>
#include <vector> // For std::vector usage

namespace cms::alpakatools {

  template <class T>
  struct SimpleVectorOfVectors {
    constexpr SimpleVectorOfVectors() = default;

    // Ownership of m_data stays within the caller
    constexpr void construct(int outerCapacity, int innerCapacity, T *data) {
      m_size = 0;
      m_outerCapacity = outerCapacity;
      m_innerCapacity = innerCapacity;
      m_data = data;

      // Pre-allocate space for outer vector
      m_outerIndices.reserve(outerCapacity);
    }

    // Add a new vector to the outer structure
    constexpr int add_vector() {
      if (m_size >= m_outerCapacity) {
        return -1;
      }

      // Set up indices for the new vector
      int index = m_size;
      m_outerIndices.push_back(index * m_innerCapacity);
      ++m_size;
      return index;
    }

    // Push an element into a specific inner vector (thread-safe version)
    template <typename TAcc>
    ALPAKA_FN_ACC int push_back(const TAcc &acc, int vectorIndex, const T &element) {
      if (vectorIndex >= m_size) {
        return -1; // Invalid vector index
      }

      int innerStart = m_outerIndices[vectorIndex];
      int &innerSize = m_innerSizes[vectorIndex];

      auto previousInnerSize = alpaka::atomicAdd(acc, &innerSize, 1, alpaka::hierarchy::Blocks{});
      if (previousInnerSize < m_innerCapacity) {
        m_data[innerStart + previousInnerSize] = element;
        return previousInnerSize;
      }

      alpaka::atomicSub(acc, &innerSize, 1, alpaka::hierarchy::Blocks{});
      return -1; // Inner vector full
    }

    template <typename TAcc, class... Ts>
    ALPAKA_FN_ACC int emplace_back(const TAcc &acc, int vectorIndex, Ts &&...args) {
    if (vectorIndex >= m_size) {
        return -1; // Invalid vector index
    }

    int innerStart = m_outerIndices[vectorIndex];
    int &innerSize = m_innerSizes[vectorIndex];

    // Atomically increment the inner size
    auto previousInnerSize = alpaka::atomicAdd(acc, &innerSize, 1, alpaka::hierarchy::Blocks{});

    if (previousInnerSize < m_innerCapacity) {
        // Safely construct the new object in-place
        new (&m_data[innerStart + previousInnerSize]) T(std::forward<Ts>(args)...);
        return previousInnerSize;
    } else {
        // Roll back if capacity is exceeded
        alpaka::atomicSub(acc, &innerSize, 1, alpaka::hierarchy::Blocks{});
        return -1;
    }
    }

    // Accessors for inner vectors
    inline constexpr T *get_inner_vector(int vectorIndex) {
      if (vectorIndex >= m_size) {
        return nullptr; // Invalid vector index
      }

      int innerStart = m_outerIndices[vectorIndex];
      return &m_data[innerStart];
    }

    inline constexpr int get_inner_size(int vectorIndex) const {
      if (vectorIndex >= m_size) {
        return -1; // Invalid vector index
      }

      return m_innerSizes[vectorIndex];
    }

    inline constexpr bool empty() const { return m_size == 0; }
    inline constexpr bool full() const { return m_size >= m_outerCapacity; }
    inline constexpr int size() const { return m_size; }
    inline constexpr int capacity() const { return m_outerCapacity; }
    inline constexpr int inner_capacity() const { return m_innerCapacity; }

  private:
    int m_size = 0;               // Number of outer vectors
    int m_outerCapacity = 0;      // Capacity of outer vector
    int m_innerCapacity = 0;      // Capacity of each inner vector
    T *m_data = nullptr;          // Raw data storage
    std::vector<int> m_outerIndices; // Starting indices of each inner vector
    std::vector<int> m_innerSizes;   // Sizes of each inner vector
  };

  // Factory function to create a SimpleVectorOfVectors
  template <class T>
  SimpleVectorOfVectors<T> make_SimpleVectorOfVectors(int outerCapacity, int innerCapacity, T *data) {
    SimpleVectorOfVectors<T> ret;
    ret.construct(outerCapacity, innerCapacity, data);
    return ret;
  }

  // Factory function to create a SimpleVectorOfVectors in a pre-allocated memory block
  template <class T>
  SimpleVectorOfVectors<T> *make_SimpleVectorOfVectors(SimpleVectorOfVectors<T> *mem, int outerCapacity, int innerCapacity, T *data) {
    auto ret = new (mem) SimpleVectorOfVectors<T>();
    ret->construct(outerCapacity, innerCapacity, data);
    return ret;
  }

}  // namespace cms::alpakatools
