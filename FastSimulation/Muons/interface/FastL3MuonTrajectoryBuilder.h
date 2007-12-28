#ifndef FastSimulation_Muons_FastL3MuonTrajectoryBuilder_H
#define FastSimulation_Muons_FastL3MuonTrajectoryBuilder_H

/** \class FastL3MuonTrajectoryBuilder
 *  class to build muon trajectory from STA L2 muons and tracker tracks
 *
 *  $Date: 2007/12/28 11:32:54 $
 *  $Revision: 1.0 $
 *
 *  \author Patrick Janot - CERN 
 */

#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonServiceProxy;
class Trajectory;
class TrackerSeedGenerator;
class TrajectoryCleaner;

class FastL3MuonTrajectoryBuilder : public GlobalTrajectoryBuilderBase {

  public:

    /// constructor with Parameter Set and MuonServiceProxy
    FastL3MuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*);
          
    /// destructor
    ~FastL3MuonTrajectoryBuilder();

    /// reconstruct trajectories from standalone and tracker only Tracks    
    MuonTrajectoryBuilder::CandidateContainer trajectories(const TrackCand&);

    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

  private:
  
    /// make a TrackCand collection using tracker Track, Trajectory information
    std::vector<TrackCand> makeTkCandCollection(const TrackCand&) const;

    /// build a tracker Trajectory from a seed
    TC makeTrajsFromSeeds(const std::vector<TrajectorySeed>&) const;

  private:
  
    std::vector<TrackCand> regionalTkTracks;
    TrackCand dummyStaCand;

    bool theTkTrajsAvailableFlag;
    bool theFirstEvent;
    
    TrackerSeedGenerator* theTkSeedGenerator;
    TrajectoryCleaner* theTrajectoryCleaner;

    std::string theTkBuilderName;
    edm::ESHandle<TrajectoryBuilder> theTkBuilder;

    edm::InputTag theTrackerTrajectoryCollection;
    edm::InputTag theSimModule;
    const edm::Event* theEvent;
};
#endif


