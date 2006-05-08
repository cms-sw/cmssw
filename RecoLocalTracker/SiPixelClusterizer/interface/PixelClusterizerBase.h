#ifndef RecoLocalTracker_SiPixelClusterizer_PixelClusterizerBase_H
#define RecoLocalTracker_SiPixelClusterizer_PixelClusterizerBase_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include <vector>

class PixelGeomDetUnit;

/**
 * Abstract interface for Pixel Clusterizers
 */
class PixelClusterizerBase {
public:
  typedef edm::DetSet<PixelDigi>::const_iterator    DigiIterator;

  // Virtual destructor, this is a base class.
  virtual ~PixelClusterizerBase() {}

  // New interface with DetSetVector data container

  virtual std::vector<SiPixelCluster>  
    clusterizeDetUnit( const edm::DetSet<PixelDigi> & input,	
		       const PixelGeomDetUnit * pixDet,
		       const std::vector<short>& badChannels) = 0;

  // TO DO: the way we pass bad channels is most likely have to change later.

};

#endif
