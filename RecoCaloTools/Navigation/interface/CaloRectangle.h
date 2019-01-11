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
    const int ixMin;
    const int ixMax;
    const int iyMin;
    const int iyMax;

    template<class T>
    auto operator()(T home, CaloTopology const& topology);

};

inline auto makeCaloRectangle(int size) {
    return CaloRectangle{-size, size, -size, size};
}

inline auto makeCaloRectangle(int sizeX, int sizeY) {
    return CaloRectangle{-sizeX, sizeX, -sizeY, sizeY};
}

template<class T>
T offsetBy(T start, CaloSubdetectorTopology const& topo, int dX, int dY)
{
    for(int x = 0; x < std::abs(dX) && start != T(0); x++) {
        // east is eta in barrel
        start = dX > 0 ? topo.goEast(start) : topo.goWest(start);
    }

    for(int y = 0; y < std::abs(dY) && start != T(0); y++) {
        // north is phi in barrel
        start = dY > 0 ? topo.goNorth(start) : topo.goSouth(start);
    }
    return start;
}

template<class T>
class CaloRectangleRange {

  public:

    class Iterator {

      public:

        Iterator(T const& home, int ix, int iy, CaloRectangle const rectangle, CaloSubdetectorTopology const& topology)
          : home_(home)
          , rectangle_(rectangle)
          , topology_(topology)
          , ix_(ix)
          , iy_(iy)
        {}

        Iterator& operator++() {
            if(iy_ == rectangle_.iyMax) {
                iy_ = rectangle_.iyMin;
                ix_++;
            } else ++iy_;
            return *this;
        }

        int ix() const { return ix_; }
        int iy() const { return iy_; }

        bool operator==(Iterator const& other) const { return ix_ == other.ix() && iy_ == other.iy(); }
        bool operator!=(Iterator const& other) const { return ix_ != other.ix() || iy_ != other.iy(); }

        T operator*() const { return offsetBy(home_, topology_, ix_, iy_); }

      private:

        const T home_;

        const CaloRectangle rectangle_;
        CaloSubdetectorTopology const& topology_;

        int ix_;
        int iy_;
    };

  public:
    CaloRectangleRange(CaloRectangle rectangle, T home, CaloTopology const& topology)
      : home_(home)
      , rectangle_(rectangle)
      , topology_(*topology.getSubdetectorTopology(home))
    {}

    auto begin() {
        return Iterator(home_, rectangle_.ixMin, rectangle_.iyMin, rectangle_, topology_);
    }
    auto end() {
        return Iterator(home_, rectangle_.ixMax + 1, rectangle_.iyMin, rectangle_, topology_);
    }

  private:
    const T home_;
    const CaloRectangle rectangle_;
    CaloSubdetectorTopology const& topology_;
};

template<class T>
auto CaloRectangle::operator()(T home, CaloTopology const& topology) {
    return CaloRectangleRange<T>(*this, home, topology);
}

#endif
