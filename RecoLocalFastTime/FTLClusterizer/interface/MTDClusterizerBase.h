#ifndef RecoLocalTracker_MTDClusterizer_MTDClusterizerBase_H
#define RecoLocalTracker_MTDClusterizer_MTDClusterizerBase_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include <vector>
#include <array>

/**
 * Abstract interface for MTD Clusterizers
 */
class MTDClusterizerBase {
public:
  typedef FTLRecHitCollection::const_iterator RecHitIterator;
  typedef FTLClusterCollection::const_iterator ClusterIterator;

  // Virtual destructor, this is a base class.
  virtual ~MTDClusterizerBase() {}

  // Build clusters
  virtual void clusterize(const FTLRecHitCollection& input,
                          const MTDGeometry* geom,
                          const MTDTopology* topo,
                          FTLClusterCollection& output) = 0;

protected:
  struct AccretionCluster {
    typedef unsigned short UShort;
    static constexpr UShort MAXSIZE = 256;

    std::array<float, MAXSIZE> energy;
    std::array<float, MAXSIZE> time;
    std::array<float, MAXSIZE> timeError;
    std::array<UShort, MAXSIZE> x;
    std::array<UShort, MAXSIZE> y;

    UShort xmin = 16000;
    UShort ymin = 16000;
    unsigned int isize = 0;
    unsigned int curr = 0;

    // stack interface (unsafe ok for use below)
    UShort top() const { return curr; }
    void pop() { ++curr; }
    bool empty() { return curr == isize; }

    bool add(FTLCluster::FTLHitPos const& p, float const ienergy, float const itime, float const itimeError) {
      if (isize == MAXSIZE)
        return false;
      xmin = std::min(xmin, (unsigned short)(p.row()));
      ymin = std::min(ymin, (unsigned short)(p.col()));
      energy[isize] = ienergy;
      time[isize] = itime;
      timeError[isize] = itimeError;
      x[isize] = p.row();
      y[isize] = p.col();
      isize++;
      return true;
    }
  };
};

#endif
