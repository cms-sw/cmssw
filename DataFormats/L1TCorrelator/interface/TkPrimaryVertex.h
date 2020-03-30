#ifndef TkTrigger_TkPrimaryVertex_h
#define TkTrigger_TkPrimaryVertex_h

// Nov 12, 2013
// First version of a class for L1-zvertex

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

#include <vector>

  typedef std::vector<TkPrimaryVertex> TkPrimaryVertexCollection;

}  // namespace l1t
#endif
