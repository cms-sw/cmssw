#ifndef FWCore_Utilities_IndexSet_h
#define FWCore_Utilities_IndexSet_h

#include <vector>

namespace edm {
  /**
   * A simple class representing a set of indices to another container
   * for fast insert and search.
   *
   * This class can be useful if one needs to record the indices of
   * objects that form a subset of a container (e.g. passing some
   * selection), and then repeatedly check if an object of a given
   * index belongs to that subset. As the elements are assumed to be
   * indices, the set can be implemented as a vector<bool> such that
   * each possible element corresponds an index in the vector. Then
   * the insert and search are (almost) array access operations.
   *
   * From the set opreations, only insertion, search, and clear are
   * supported for now. More can be added if needed.
   */
  class IndexSet {
  public:
    /// Construct empty set
    IndexSet() : numTrueElements_(0) {}

    /// Check if the set is empty
    bool empty() const { return numTrueElements_ == 0; }

    /// Number of elements in the set
    unsigned int size() const { return numTrueElements_; }

    /// Reserve memory for the set
    void reserve(unsigned int size) {
      if (size >= content_.size()) {
        content_.resize(size, false);
      }
    }

    /// Clear the set
    void clear() {
      std::fill(begin(content_), end(content_), false);
      numTrueElements_ = 0;
    }

    /// Insert an element (=index) to the set
    void insert(unsigned int index) {
      reserve(index + 1);
      numTrueElements_ += !content_[index];  // count increases if value was false
      content_[index] = true;
    }

    /// Check if an element (=index) is in the set
    bool has(unsigned int index) const { return index < content_.size() && content_[index]; }

  private:
    std::vector<bool>
        content_;  /// Each possible element of the set corresponds an index in this vector. The value of an element tells if that index exists in the set (true) or not (false).
    unsigned int numTrueElements_;  /// Count of true elements is equivalent of "size()" of std::set
  };
}  // namespace edm

#endif
