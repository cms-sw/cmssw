#ifndef CalibTracker_SiSitripLorentzAngle_SiStripFineDelayTLA_h
#define CalibTracker_SiSitripLorentzAngle_SiStripFineDelayTLA_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DataFormats/TrackingRecHit/interface/TrackingRecHit.h>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class TrackerGeometry;
class TrackingRecHit;
class TrajectorySeed;
class Trajectory;

class SiStripFineDelayTLA {
public:
  explicit SiStripFineDelayTLA(const edm::ParameterSet& conf, edm::ConsumesCollector iC);
  virtual ~SiStripFineDelayTLA();
  void init(const edm::Event& e, const edm::EventSetup& c);

  std::vector<std::pair<std::pair<DetId, LocalPoint>, float> > findtrackangle(const std::vector<Trajectory>& traj);
  std::vector<std::pair<std::pair<DetId, LocalPoint>, float> > findtrackangle(const Trajectory& traj);

private:
  double computeAngleCorr(const LocalVector& v, double pitch, double thickness);

private:
  edm::ParameterSet conf_;
  const TrackerGeometry* tracker;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
};

#endif
