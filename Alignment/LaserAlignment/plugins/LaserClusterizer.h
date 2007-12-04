#ifndef LaserAlignment_LaserClusterizer_h
#define LaserAlignment_LaserClusterizer_h

/** \class LaserClusterizer
 *  Clusterizer for the Laser beams
 *
 *  $Date: 2007/03/18 19:00:20 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


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
