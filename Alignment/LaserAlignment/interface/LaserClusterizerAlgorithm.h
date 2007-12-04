#ifndef LaserAlignment_LaserClusterizerAlgorithm_h
#define LaserAlignment_LaserClusterizerAlgorithm_h

/** \class LaserClusterizerAlgorithm
 *  the Clusterizer Algorithm for the laser beams
 *
 *  $Date: 2007/03/18 19:00:19 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


class LaserBeamClusterizer;

class LaserClusterizerAlgorithm
{
 public:
	/// constructor
  LaserClusterizerAlgorithm(const edm::ParameterSet & theConf);
	/// destructor
  ~LaserClusterizerAlgorithm();

  /// Runs the Algorithm
  void run(const edm::DetSetVector<SiStripDigi>& input, const LASBeamProfileFitCollection* beamFit,
	   edm::DetSetVector<SiStripCluster>& output, const edm::ESHandle<TrackerGeometry>& theTrackerGeometry);

 private:
  edm::ParameterSet theConfiguration;
  LaserBeamClusterizer * theBeamClusterizer;
  std::string theClusterMode;
  // width of the cluster in sigma's
  double theClusterWidth;
  bool theValidClusterizer;
};
#endif
