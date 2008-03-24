#ifndef FastSimulation_Muons_FastL3MuonTrajectoryBuilder_H
#define FastSimulation_Muons_FastL3MuonTrajectoryBuilder_H

/** \class FastL3MuonTrajectoryBuilder
 *  class to build muon trajectory from STA L2 muons and tracker tracks
 *
 *  $Date: 2008/03/14 19:12:06 $
 *  $Revision: 1.4 $
 *
 *  \author Patrick Janot - CERN 
 */

//for debug only 
//#define FAMOS_DEBUG

#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#ifdef FAMOS_DEBUG
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#endif

namespace edm {
  class ParameterSet; 
  class Event; 
  class EventSetup;
}

namespace reco { 
  class Track;
}

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

    /// clear memory
    void clear();

  private:
  
    /// make a TrackCand collection using tracker Track, Trajectory information
    std::vector<TrackCand> makeTkCandCollection(const TrackCand&);

    /// build a tracker Trajectory from a seed
    TC makeTrajsFromSeeds(const std::vector<TrajectorySeed>&) const;

    /// Find the simTrack id of a reco track
    int findId(const reco::Track& aTrack) const;

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

#ifdef FAMOS_DEBUG
    DQMStore * dbe;
    MonitorElement* simuMuons;
    MonitorElement* matchMuons;
    MonitorElement* refitMuons;
#endif

};
#endif


