#ifndef RecoLocalTracker_Phase2ITPixelClusterizer_Phase2ITPixelClusterizerBase_H
#define RecoLocalTracker_Phase2ITPixelClusterizer_Phase2ITPixelClusterizerBase_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2ITPixelCluster/interface/Phase2ITPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"
#include <vector>

class PixelGeomDetUnit;

/**
 * Abstract interface for Pixel Clusterizers
 */
class Phase2ITPixelClusterizerBase {
public:
  typedef edm::DetSet<PixelDigi>::const_iterator    DigiIterator;

  // Virtual destructor, this is a base class.
  virtual ~Phase2ITPixelClusterizerBase() {}

  // Build clusters in a DetUnit. Both digi and cluster stored in a DetSet

  virtual void clusterizeDetUnit( const edm::DetSet<PixelDigi> & input,	
				  const PixelGeomDetUnit * pixDet,
				  const std::vector<short>& badChannels,
				  edmNew::DetSetVector<Phase2ITPixelCluster>::FastFiller& output) = 0;

  // Configure gain calibration service
  void setSiPixelGainCalibrationService( SiPixelGainCalibrationServiceBase* in){ 
    theSiPixelGainCalibrationService_=in;
  }

 protected:
  SiPixelGainCalibrationServiceBase* theSiPixelGainCalibrationService_;

};

#endif
