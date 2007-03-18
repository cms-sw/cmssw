#ifndef LaserAlignment_LaserBeamClusterizer_h
#define LaserAlignment_LaserBeamClusterizer_h

/** \class LaserBeamClusterizer
 *  uses the results of the BeamProfileFit to create SiStripClusters from the StripDigis of the Laser beams
 *
 *  $Date: Fri Mar 16 15:51:33 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

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

	/// constructor
  LaserBeamClusterizer() {};
	/// destructor
  ~LaserBeamClusterizer() {};

	/// do the clusterizing
  void clusterizeDetUnit(const edm::DetSet<SiStripDigi>& input, edm::DetSet<SiStripCluster>& output,
			 BeamFitIterator beginFit, BeamFitIterator endFit,
			 unsigned int detId, double ClusterWidth);
};
#endif

