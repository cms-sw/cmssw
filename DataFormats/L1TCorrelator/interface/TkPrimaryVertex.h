#ifndef DataFormatsL1TCorrelator_TkPrimaryVertex_h
#define DataFormatsL1TCorrelator_TkPrimaryVertex_h

// Package:     L1TCorrelator
// Class  :     TkPrimaryVertex

// First version of a class for L1-zvertex
#include <vector>

namespace l1t {

  class TkPrimaryVertex {
  public:
    TkPrimaryVertex() : zvertex_(-999), sum_(-999) {}

    ~TkPrimaryVertex() {}

    TkPrimaryVertex(float z, float s) : zvertex_(z), sum_(s) {}

    float zvertex() const { return zvertex_; }
    float sum() const { return sum_; }

  private:
    float zvertex_;
    float sum_;
  };

  typedef std::vector<TkPrimaryVertex> TkPrimaryVertexCollection;

}  // namespace l1t
#endif
