#ifndef RecoCaloTools_Navigation_CaloRectangle_H
#define RecoCaloTools_Navigation_CaloRectangle_H

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

/*
 * CaloRectangle is a class to create a rectangular range of DetIds around a
 * central DetId. Meant to be used for range-based loops to calculate cluster
 * shape variables.
 */

struct CaloRectangle {
  const int iEtaOrIXMin;
  const int iEtaOrIXMax;
  const int iPhiOrIYMin;
  const int iPhiOrIYMax;

  template <class T>
  auto operator()(T home, CaloTopology const& topology);
};

template <class T>
T offsetBy(T start, CaloSubdetectorTopology const* topo, int dIEtaOrIX, int dIPhiOrIY) {
  if (topo) {
    for (int i = 0; i < std::abs(dIEtaOrIX) && start != T(0); i++) {
      start = dIEtaOrIX > 0 ? topo->goEast(start) : topo->goWest(start);
    }

    for (int i = 0; i < std::abs(dIPhiOrIY) && start != T(0); i++) {
      start = dIPhiOrIY > 0 ? topo->goNorth(start) : topo->goSouth(start);
    }
  }
  return start;
}

template <class T>
class CaloRectangleRange {
public:
  class Iterator {
  public:
    Iterator(T const& home,
             int iEtaOrIX,
             int iPhiOrIY,
             CaloRectangle const rectangle,
             CaloSubdetectorTopology const* topology)
        : home_(home), rectangle_(rectangle), topology_(topology), iEtaOrIX_(iEtaOrIX), iPhiOrIY_(iPhiOrIY) {}

    Iterator& operator++() {
      if (iPhiOrIY_ == rectangle_.iPhiOrIYMax) {
        iPhiOrIY_ = rectangle_.iPhiOrIYMin;
        iEtaOrIX_++;
      } else
        ++iPhiOrIY_;
      return *this;
    }

    int iEtaOrIX() const { return iEtaOrIX_; }
    int iPhiOrIY() const { return iPhiOrIY_; }

    bool operator==(Iterator const& other) const {
      return iEtaOrIX_ == other.iEtaOrIX() && iPhiOrIY_ == other.iPhiOrIY();
    }
    bool operator!=(Iterator const& other) const {
      return iEtaOrIX_ != other.iEtaOrIX() || iPhiOrIY_ != other.iPhiOrIY();
    }

    T operator*() const { return offsetBy(home_, topology_, iEtaOrIX_, iPhiOrIY_); }

  private:
    const T home_;

    const CaloRectangle rectangle_;
    CaloSubdetectorTopology const* topology_;

    int iEtaOrIX_;
    int iPhiOrIY_;
  };

public:
  CaloRectangleRange(CaloRectangle rectangle, T home, CaloTopology const& topology)
      : home_(home), rectangle_(rectangle), topology_(topology.getSubdetectorTopology(home)) {}

  CaloRectangleRange(int size, T home, CaloTopology const& topology)
      : home_(home), rectangle_{-size, size, -size, size}, topology_(topology.getSubdetectorTopology(home)) {}

  auto begin() { return Iterator(home_, rectangle_.iEtaOrIXMin, rectangle_.iPhiOrIYMin, rectangle_, topology_); }
  auto end() { return Iterator(home_, rectangle_.iEtaOrIXMax + 1, rectangle_.iPhiOrIYMin, rectangle_, topology_); }

private:
  const T home_;
  const CaloRectangle rectangle_;
  CaloSubdetectorTopology const* topology_;
};

template <class T>
auto CaloRectangle::operator()(T home, CaloTopology const& topology) {
  return CaloRectangleRange<T>(*this, home, topology);
}

#endif
