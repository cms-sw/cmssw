#ifndef LaserAlignment_LaserClusterizerAlgorithm_h
#define LaserAlignment_LaserClusterizerAlgorithm_h

/** \class LaserClusterizerAlgorithm
 *  the Clusterizer Algorithm for the laser beams
 *
 *  $Date: Fri Mar 16 15:56:29 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Alignment/LaserAlignment/interface/LaserBeamClusterizer.h"

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
