#ifndef CosmicMuonTrajectoryBuilder_H
#define CosmicMuonTrajectoryBuilder_H
/** \file CosmicMuonTrajectoryBuilder
 *
 *  $Date: 2006/07/26 17:46:48 $
 *  $Revision: 1.4 $
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

   // fake implementation 
   // return a container reconstructed muons starting from a given track
  virtual CandidateContainer trajectories(const reco::TrackRef&) {
    return CandidateContainer();
  }

  virtual void setEvent(const edm::Event&);

  const Propagator* propagator() const {return &*theService->propagator(thePropagatorName);}
  MuonTrajectoryUpdator* updator() const {return theUpdator;}

private:

  MuonTrajectoryUpdator* theUpdator;
  MuonDetLayerMeasurements* theLayerMeasurements;

  const MuonServiceProxy *theService;

  std::string thePropagatorName;
  
};
#endif
