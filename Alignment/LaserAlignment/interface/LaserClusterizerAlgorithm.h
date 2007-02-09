/* 
 * the Clusterizer Algorithm for the laser beams
 */

#ifndef LaserAlignment_LaserClusterizerAlgorithm_h
#define LaserAlignment_LaserClusterizerAlgorithm_h

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
  LaserClusterizerAlgorithm(const edm::ParameterSet & theConf);
  ~LaserClusterizerAlgorithm();

  // Runs the Algorithm
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
