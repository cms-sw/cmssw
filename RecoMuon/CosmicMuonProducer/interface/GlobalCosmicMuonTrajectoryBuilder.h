#ifndef RecoMuon_CosmicMuonProducer_GlobalCosmicMuonTrajectoryBuilder_H
#define RecoMuon_CosmicMuonProducer_GlobalCosmicMuonTrajectoryBuilder_H

/** \file GlobalCosmicMuonTrajectoryBuilder
 *  class to build combined trajectory from cosmic tracks in tk and mu
 *
 *  $Date: 2006/11/03 20:00:18 $
 *  $Revision: 1.3 $
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackReFitter.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackConverter.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class Trajectory;
class TrajectoryMeasurement;

class GlobalCosmicMuonTrajectoryBuilder : public MuonTrajectoryBuilder{

public:
  typedef TransientTrackingRecHit::RecHitContainer RecHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
  typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
  typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;


  GlobalCosmicMuonTrajectoryBuilder(const edm::ParameterSet&,const MuonServiceProxy* service);
  virtual ~GlobalCosmicMuonTrajectoryBuilder();

  /// dummy implementation, unused in this class
  std::vector<Trajectory*> trajectories(const TrajectorySeed&) {return std::vector<Trajectory*>();}

  /// choose tk Track and build combined trajectories
  virtual CandidateContainer trajectories(const TrackCand&);


  virtual void setEvent(const edm::Event&);

  std::pair<bool,double> match(const reco::Track&, const reco::Track&);

private:

  const MuonServiceProxy *theService;

  MuonTrackReFitter* theRefitter;
  MuonTrackConverter* theTrackConverter;

  struct DecreasingGlobalY{
    bool operator()(const TransientTrackingRecHit::ConstRecHitPointer &lhs,
		    const TransientTrackingRecHit::ConstRecHitPointer &rhs) const{ 
      return lhs->globalPosition().y() > rhs->globalPosition().y(); 
    }
  };

  std::string thePropagatorName;
  std::string theTkTrackLabel;
  std::string theTTRHBuilderName;

  edm::Handle<reco::TrackCollection> theTrackerTracks;

  bool tkTrajsAvailable;

  const std::vector<Trajectory>* allTrackerTrajs;
  
};
#endif
