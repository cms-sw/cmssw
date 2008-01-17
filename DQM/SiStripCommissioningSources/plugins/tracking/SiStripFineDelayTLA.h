#ifndef CalibTracker_SiSitripLorentzAngle_SiStripFineDelayTLA_h
#define CalibTracker_SiSitripLorentzAngle_SiStripFineDelayTLA_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DataFormats/TrackingRecHit/interface/TrackingRecHit.h>
#include "DataFormats/TrackReco/interface/Track.h"

class SimpleTrackRefitter;
class TrackerGeometry;
class TrackingRecHit;
class TrajectorySeed;
class Trajectory;

class SiStripFineDelayTLA 
{
 public:
  
  explicit SiStripFineDelayTLA(const edm::ParameterSet& conf);
  virtual ~SiStripFineDelayTLA();
  void init(const edm::Event& e,const edm::EventSetup& c);

  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > findtrackangle(const TrajectorySeed& seed,const reco::Track & theT);
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > findtrackangle(const reco::Track & theT);
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > findtrackangle(const std::vector<Trajectory>& traj);
  std::vector<std::pair< std::pair<DetId, LocalPoint> ,float> > findtrackangle(const Trajectory& traj);

 private:
  edm::ParameterSet conf_;
  SimpleTrackRefitter* refitter_;
  const TrackerGeometry * tracker;
};


#endif
