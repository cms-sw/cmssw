#ifndef RecoLocalTracker_SiPixelClusterizer_PixelClusterizerBase_H
#define RecoLocalTracker_SiPixelClusterizer_PixelClusterizerBase_H

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include <vector>

/**
 * Abstract interface for Pixel Clusterizers
 */
class PixelClusterizerBase {
public:
  // Define typedefs to simplify porting from ORCA.
  typedef std::vector<PixelDigi>                    DigiContainer;
  typedef DigiContainer::const_iterator             DigiIterator;

  // Virtual destructor, this is a base class.
  virtual ~PixelClusterizerBase() {}

  // Clusterize one contiguous chunk of silicon (a panel or a plaquette)
  virtual 
    std::vector<SiPixelCluster>  
    clusterizeDetUnit( DigiIterator begin, DigiIterator end,
		       unsigned int detid,
		       const std::vector<float>& noiseVec,
		       const std::vector<short>& badChannels) = 0;
  // TO DO: the way we pass noise and bad channels is most likely
  //        have to change later.

};

#endif
