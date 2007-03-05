/*
 * Clusterizer for the Laser beams
 */

#ifndef LaserAlignment_LaserClusterizer_h
#define LaserAlignment_LaserClusterizer_h

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

  explicit LaserClusterizer(const edm::ParameterSet & theConf);
  virtual ~LaserClusterizer();

  virtual void beginJob(const edm::EventSetup& theSetup);

  virtual void produce(edm::Event& theEvent, const edm::EventSetup& theSetup);

 private:
  LaserClusterizerAlgorithm theLaserClusterizerAlgorithm;
  edm::ParameterSet theParameterSet;
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
};
#endif
