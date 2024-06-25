// Author: Felice Pantaleo (CERN), 2023, felice.pantaleo@cern.ch
#ifndef MultiVectorManager_h
#define MultiVectorManager_h

#include <vector>
#include <cassert>
#include <algorithm>
#include <iterator>

template <typename T>
class MultiVectorManager {
public:
  void addVector(const std::vector<T>& vec) {
    vectors.emplace_back(vec.begin(), vec.end());
    offsets.push_back(totalSize);
    totalSize += vec.size();
  }

  T& operator[](size_t globalIndex) {
    return const_cast<T&>(static_cast<const MultiVectorManager*>(this)->operator[](globalIndex));
  }

  const T& operator[](size_t globalIndex) const {
    assert(globalIndex < totalSize && "Global index out of range");

    auto it = std::upper_bound(offsets.begin(), offsets.end(), globalIndex);
    size_t vectorIndex = std::distance(offsets.begin(), it) - 1;
    size_t localIndex = globalIndex - offsets[vectorIndex];

    return vectors[vectorIndex][localIndex];
  }

  size_t getGlobalIndex(size_t vectorIndex, size_t localIndex) const {
    assert(vectorIndex < vectors.size() && "Vector index out of range");

    const auto& vec = vectors[vectorIndex];
    assert(localIndex < vec.size() && "Local index out of range");

    return offsets[vectorIndex] + localIndex;
  }

  size_t size() const { return totalSize; }

  std::pair<size_t, size_t> getVectorAndLocalIndex(size_t globalIndex) const {
    assert(globalIndex < totalSize && "Global index out of range");

    auto it = std::upper_bound(offsets.begin(), offsets.end(), globalIndex);
    size_t vectorIndex = std::distance(offsets.begin(), it) - 1;
    size_t localIndex = globalIndex - offsets[vectorIndex];

    return {vectorIndex, localIndex};
  }

  class Iterator {
  public:
    Iterator(const MultiVectorManager& manager, size_t index) : manager(manager), currentIndex(index) {}

    bool operator!=(const Iterator& other) const { return currentIndex != other.currentIndex; }

    T& operator*() const { return const_cast<T&>(manager[currentIndex]); }

    void operator++() { ++currentIndex; }

  private:
    const MultiVectorManager& manager;
    size_t currentIndex;
  };

  Iterator begin() const { return Iterator(*this, 0); }

  Iterator end() const { return Iterator(*this, totalSize); }

private:
  std::vector<std::vector<T>> vectors;
  std::vector<size_t> offsets;
  size_t totalSize = 0;
};

#endif
