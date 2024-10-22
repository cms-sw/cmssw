#ifndef RecoLocalTracker_SiPixelClusterizer_PixelClusterizerBase_H
#define RecoLocalTracker_SiPixelClusterizer_PixelClusterizerBase_H

#include <vector>

#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

class PixelGeomDetUnit;

/**
 * Abstract interface for Pixel Clusterizers
 */
class PixelClusterizerBase {
public:
  typedef edm::DetSet<PixelDigi>::const_iterator DigiIterator;
  typedef edmNew::DetSet<SiPixelCluster>::const_iterator ClusterIterator;

  struct AccretionCluster {
    static constexpr uint16_t MAXSIZE = 256;
    uint16_t adc[MAXSIZE];
    uint16_t x[MAXSIZE];
    uint16_t y[MAXSIZE];
    uint16_t xmin = 16000;
    uint16_t ymin = 16000;
    unsigned int isize = 0;
    int charge = 0;

    // stack interface (unsafe ok for use below)
    unsigned int curr = 0;
    uint16_t top() const { return curr; }
    void pop() { ++curr; }
    bool empty() { return curr == isize; }

    void clear() {
      xmin = 16000;
      ymin = 16000;
      isize = 0;
      charge = 0;
      curr = 0;
    }

    bool add(SiPixelCluster::PixelPos const& p, uint16_t const iadc) {
      if (isize == MAXSIZE)
        return false;
      xmin = std::min<uint16_t>(xmin, p.row());
      ymin = std::min<uint16_t>(ymin, p.col());
      adc[isize] = iadc;
      x[isize] = p.row();
      y[isize++] = p.col();
      charge += iadc;
      return true;
    }
  };

  // Virtual destructor, this is a base class.
  virtual ~PixelClusterizerBase() {}

  // Build clusters in a DetUnit. Both digi and cluster stored in a DetSet

  virtual void clusterizeDetUnit(const edm::DetSet<PixelDigi>& input,
                                 const PixelGeomDetUnit* pixDet,
                                 const TrackerTopology* tTopo,
                                 const std::vector<short>& badChannels,
                                 edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) = 0;

  virtual void clusterizeDetUnit(const edmNew::DetSet<SiPixelCluster>& input,
                                 const PixelGeomDetUnit* pixDet,
                                 const TrackerTopology* tTopo,
                                 const std::vector<short>& badChannels,
                                 edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) = 0;

  // Configure gain calibration service
  void setSiPixelGainCalibrationService(SiPixelGainCalibrationServiceBase* in) {
    theSiPixelGainCalibrationService_ = in;
  }

protected:
  SiPixelGainCalibrationServiceBase* theSiPixelGainCalibrationService_;
};

#endif
