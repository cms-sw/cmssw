#ifndef LaserAlignment_LaserClusterizer_h
#define LaserAlignment_LaserClusterizer_h

/** \class LaserClusterizer
 *  Clusterizer for the Laser beams
 *
 *  $Date: Sun Mar 18 19:43:53 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/LaserAlignment/interface/LASBeamProfileFitCollection.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "Alignment/LaserAlignment/interface/LaserClusterizerAlgorithm.h"

class LaserClusterizer : public edm::EDProducer
{
 public:
  typedef std::vector<edm::ParameterSet> Parameters;

	/// constructor
  explicit LaserClusterizer(const edm::ParameterSet & theConf);
	/// destructor
  virtual ~LaserClusterizer();

	/// begin job
  virtual void beginJob(const edm::EventSetup& theSetup);
	/// produce clusters from the laser beams
  virtual void produce(edm::Event& theEvent, const edm::EventSetup& theSetup);

 private:
  LaserClusterizerAlgorithm theLaserClusterizerAlgorithm;
  edm::ParameterSet theParameterSet;
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
};
#endif
