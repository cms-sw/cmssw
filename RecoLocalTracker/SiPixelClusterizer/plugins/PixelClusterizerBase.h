#ifndef RecoLocalTracker_SiPixelClusterizer_PixelClusterizerBase_H
#define RecoLocalTracker_SiPixelClusterizer_PixelClusterizerBase_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"
#include <vector>

class PixelGeomDetUnit;

/**
 * Abstract interface for Pixel Clusterizers
 */
class PixelClusterizerBase {
public:
  typedef edm::DetSet<PixelDigi>::const_iterator    DigiIterator;
  typedef edmNew::DetSet<SiPixelCluster>::const_iterator    ClusterIterator;

  struct AccretionCluster {
    typedef unsigned short UShort;
    static constexpr UShort MAXSIZE = 256;
    UShort adc[MAXSIZE];
    UShort x[MAXSIZE];
    UShort y[MAXSIZE];
    UShort xmin=16000;
    UShort ymin=16000;
    unsigned int isize=0;
    unsigned int curr=0;

    // stack interface (unsafe ok for use below)
    UShort top() const { return curr;}
    void pop() { ++curr;}
    bool empty() { return curr==isize;}

    bool add(SiPixelCluster::PixelPos const & p, UShort const iadc) {
      if (isize==MAXSIZE) return false;
      xmin=std::min(xmin,(unsigned short)(p.row()));
      ymin=std::min(ymin,(unsigned short)(p.col()));
      adc[isize]=iadc;
      x[isize]=p.row();
      y[isize++]=p.col();
      return true;
    }
  };

  // Virtual destructor, this is a base class.
  virtual ~PixelClusterizerBase() {}

  // Build clusters in a DetUnit. Both digi and cluster stored in a DetSet

  virtual void clusterizeDetUnit( const edm::DetSet<PixelDigi> & input,	
				  const PixelGeomDetUnit * pixDet,
				  const TrackerTopology* tTopo,
				  const std::vector<short>& badChannels,
				  edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) = 0;

  virtual void clusterizeDetUnit( const edmNew::DetSet<SiPixelCluster> & input,
                                  const PixelGeomDetUnit * pixDet,
				  const TrackerTopology* tTopo,
                                  const std::vector<short>& badChannels,
                                  edmNew::DetSetVector<SiPixelCluster>::FastFiller& output) = 0;

  // Configure gain calibration service
  void setSiPixelGainCalibrationService( SiPixelGainCalibrationServiceBase* in){ 
    theSiPixelGainCalibrationService_=in;
  }

 protected:
  SiPixelGainCalibrationServiceBase* theSiPixelGainCalibrationService_;

};

#endif
