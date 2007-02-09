/* 
 * uses the results of the BeamProfileFit
 * to create SiStripClusters from the 
 * StripDigis of the Laser beams
 */


#ifndef LaserAlignment_LaserBeamClusterizer_h
#define LaserAlignment_LaserBeamClusterizer_h

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFit.h"

#include <iostream>
#include <vector>
#include <algorithm>

class LaserBeamClusterizer
{
 public: 
  typedef std::vector<LASBeamProfileFit> BeamFitContainer; 
  typedef BeamFitContainer::const_iterator BeamFitIterator;

  LaserBeamClusterizer() {};
  ~LaserBeamClusterizer() {};

  void clusterizeDetUnit(const edm::DetSet<SiStripDigi>& input, edm::DetSet<SiStripCluster>& output,
			 BeamFitIterator beginFit, BeamFitIterator endFit,
			 unsigned int detId, double ClusterWidth);
};
#endif
