#ifndef FWCore_Utilities_IndexSet_h
#define FWCore_Utilities_IndexSet_h

#include <vector>


namespace edm {
  class IndexSet {
  public:
    IndexSet(): numTrueElements_(0) {}

    bool empty() const { return numTrueElements_ == 0; }
    unsigned int size() const { return numTrueElements_; }

    void reserve(unsigned int size) {
      if(size >= content_.size()) {
        content_.resize(size, false);
      }
    }

    void clear() {
      std::fill(begin(content_), end(content_), false);
      numTrueElements_ = 0;
    }

    void insert(unsigned int index) {
      reserve(index+1);
      numTrueElements_ += !content_[index]; // count increases if value was false
      content_[index] = true;
    }

    bool has(unsigned int index) const {
      return index < content_.size() && content_[index];
    }

  private:
    std::vector<bool> content_;
    unsigned int numTrueElements_; /// Count of true elements is equivalent of "size()" of std::set
  };
}

#endif
