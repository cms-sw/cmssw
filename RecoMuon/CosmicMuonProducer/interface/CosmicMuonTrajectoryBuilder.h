#ifndef CosmicMuonTrajectoryBuilder_H
#define CosmicMuonTrajectoryBuilder_H
/** \file CosmicMuonTrajectoryBuilder
 *
 *  $Date: 2006/11/03 19:49:37 $
 *  $Revision: 1.7 $
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class Trajectory;
class TrajectoryMeasurement;
class MuonDetLayerMeasurements;

class CosmicMuonTrajectoryBuilder : public MuonTrajectoryBuilder{
public:

  CosmicMuonTrajectoryBuilder(const edm::ParameterSet&,const MuonServiceProxy* service);
  virtual ~CosmicMuonTrajectoryBuilder();

  std::vector<Trajectory*> trajectories(const TrajectorySeed&);

  /// dummy implementation, unused in this class
  virtual CandidateContainer trajectories(const TrackCand&) {
    return CandidateContainer();
  }

  virtual void setEvent(const edm::Event&);

  const Propagator* propagator() const {return &*theService->propagator(thePropagatorName);}
  MuonTrajectoryUpdator* updator() const {return theUpdator;}
  MuonTrajectoryUpdator* backwardUpdator() const {return theBKUpdator;}

private:

  MuonTrajectoryUpdator* theUpdator;
  MuonTrajectoryUpdator* theBKUpdator;
  MuonDetLayerMeasurements* theLayerMeasurements;

  const MuonServiceProxy *theService;

  std::string thePropagatorName;
  
};
#endif
