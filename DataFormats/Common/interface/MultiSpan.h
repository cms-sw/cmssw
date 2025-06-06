// Author: Felice Pantaleo (CERN), 2023, felice.pantaleo@cern.ch
#ifndef DataFormats_Common_MultiSpan_h
#define DataFormats_Common_MultiSpan_h

#include <algorithm>
#include <cassert>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

#include "DataFormats/Common/interface/RefProd.h"

namespace edm {

  /**
* @brief A view-like container that provides a contiguous indexing interface over multiple disjoint spans.
*
* MultiSpan allows to append multiple `std::vector<T>` as std::span<const T> instances and access them through a
* single global index as if they formed one continuous sequence. 
*
* This class is read-only and does not take ownership of the underlying data.
* It is intended for iteration over heterogeneous but logically connected data ranges without copying 
* or merging them into a single container.
*
* To find a span that corresponds to a global index, a binary search is used, making the access time logarithmic in the number of spans.
* This means when iterating over the elements the binary search over spans is repeated for every element.
*
*/
  template <typename T>
  class MultiSpan {
  public:
    MultiSpan() = default;

    MultiSpan(const std::vector<edm::RefProd<std::vector<T>>>& refProducts) {
      std::ranges::for_each(refProducts, [&](auto const& rp) { add(*rp); });
    }

    void add(std::span<const T> sp) {
      // Empty spans are not added to reduce the number of spans and speed up the binary search
      if (sp.empty()) {
        return;
      }

      spans_.emplace_back(sp);
      offsets_.push_back(totalSize_);
      totalSize_ += sp.size();
    }

    const T& operator[](const std::size_t globalIndex) const {
#ifndef NDEBUG
      if (globalIndex >= totalSize_) {
        throw std::out_of_range("Global index out of range");
      }
#endif
      const auto [spanIndex, indexWithinSpan] = spanAndLocalIndex(globalIndex);
      return spans_[spanIndex][indexWithinSpan];
    }

    std::size_t globalIndex(const std::size_t spanIndex, const std::size_t indexWithinSpan) const {
#ifndef NDEBUG
      if (spanIndex >= spans_.size()) {
        throw std::out_of_range("spanIndex index out of range");
      }
      if (indexWithinSpan >= spans_[spanIndex].size()) {
        throw std::out_of_range("indexWithinSpan index out of range");
      }
#endif

      return offsets_[spanIndex] + indexWithinSpan;
    }

    std::pair<std::size_t, std::size_t> spanAndLocalIndex(const std::size_t globalIndex) const {
#ifndef NDEBUG
      if (globalIndex >= totalSize_) {
        throw std::out_of_range("Global index out of range");
      }
#endif
      auto it = std::upper_bound(offsets_.begin(), offsets_.end(), globalIndex);
      std::size_t spanIndex = std::distance(offsets_.begin(), it) - 1;
      std::size_t indexWithinSpan = globalIndex - offsets_[spanIndex];

      return {spanIndex, indexWithinSpan};
    }

    std::size_t size() const { return totalSize_; }

    class ConstRandomAccessIterator {
    public:
      using iterator_category = std::random_access_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = T;
      using pointer = const T*;
      using reference = const T&;

      ConstRandomAccessIterator(const MultiSpan& ms, const std::size_t index) : ms_(&ms), currentIndex_(index) {}

      reference operator*() const { return (*ms_)[currentIndex_]; }
      pointer operator->() const { return &(*ms_)[currentIndex_]; }

      reference operator[](difference_type n) const { return (*ms_)[currentIndex_ + n]; }

      ConstRandomAccessIterator& operator++() {
        ++currentIndex_;
        return *this;
      }
      ConstRandomAccessIterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
      }
      ConstRandomAccessIterator& operator--() {
        --currentIndex_;
        return *this;
      }
      ConstRandomAccessIterator operator--(int) {
        auto tmp = *this;
        --(*this);
        return tmp;
      }

      ConstRandomAccessIterator& operator+=(difference_type n) {
        currentIndex_ += n;
        return *this;
      }
      ConstRandomAccessIterator& operator-=(difference_type n) {
        currentIndex_ -= n;
        return *this;
      }

      ConstRandomAccessIterator operator+(difference_type n) const {
        return ConstRandomAccessIterator(*ms_, currentIndex_ + n);
      }

      ConstRandomAccessIterator operator-(difference_type n) const {
        return ConstRandomAccessIterator(*ms_, currentIndex_ - n);
      }

      difference_type operator-(const ConstRandomAccessIterator& other) const {
        return currentIndex_ - other.currentIndex_;
      }

      bool operator==(const ConstRandomAccessIterator& other) const { return currentIndex_ == other.currentIndex_; }
      bool operator!=(const ConstRandomAccessIterator& other) const { return currentIndex_ != other.currentIndex_; }
      auto operator<=>(const ConstRandomAccessIterator& other) const { return currentIndex_ <=> other.currentIndex_; }

    private:
      const MultiSpan* ms_;
      std::size_t currentIndex_;
    };

    using const_iterator = ConstRandomAccessIterator;

    const_iterator begin() const { return const_iterator(*this, 0); }
    const_iterator end() const { return const_iterator(*this, totalSize_); }

  private:
    std::vector<std::span<const T>> spans_;
    std::vector<std::size_t> offsets_;
    std::size_t totalSize_{0};
  };

}  // namespace edm

#endif  // DataFormats_Common_MultiSpan_h
