#ifndef RecoMuon_L3TrackFinder_L3MuonTrajectoryBuilder_H
#define RecoMuon_L3TrackFinder_L3MuonTrajectoryBuilder_H

/** \class L3MuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  $Date:  $
 *  $Revision:  $
 *
 *  \author N. Neumeister 	 Purdue University
 *  \author C. Liu 		 Purdue University
 *  \author A. Everett 		 Purdue University
 */

#include "RecoMuon/GlobalTrackingTools/interface/GlobalTrajectoryBuilderBase.h"
//#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonServiceProxy;
class Trajectory;
class TrackerSeedGenerator;
class TrajectoryCleaner;

class L3MuonTrajectoryBuilder : public GlobalTrajectoryBuilderBase {

  public:

    /// constructor with Parameter Set and MuonServiceProxy
    L3MuonTrajectoryBuilder(const edm::ParameterSet&, const MuonServiceProxy*);
          
    /// destructor
    ~L3MuonTrajectoryBuilder();

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
  
    bool theTkTrajsAvailableFlag;
    bool theFirstEvent;
    
    TrackerSeedGenerator* theTkSeedGenerator;
    TrajectoryCleaner* theTrajectoryCleaner;

    std::string theTkBuilderName;
    edm::ESHandle<TrackerTrajectoryBuilder> theTkBuilder;

    NavigationSchool*  theNavigationSchool;
    unsigned long long theCacheId_DG;
    unsigned long long theCacheId_MG;

};
#endif
