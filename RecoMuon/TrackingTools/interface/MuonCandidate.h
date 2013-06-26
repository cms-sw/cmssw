#ifndef RecoMuon_TrackingTools_MuonCandidate_H
#define RecoMuon_TrackingTools_MuonCandidate_H

/** \class MuonCandidate
 *  Auxiliary class for muon candidates
 *
 *  $Date: 2008/04/15 12:03:09 $
 *  $Revision: 1.8 $
 *  \author N. Neumeister	Purdue University 
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>


class MuonCandidate { 
  
  public:

    typedef std::vector<Trajectory*> TrajectoryContainer; 
    typedef std::vector<MuonCandidate*> CandidateContainer;

  public:
  
    /// constructor
    MuonCandidate(Trajectory* traj, 
		  const reco::TrackRef& muon, 
		  const reco::TrackRef& tracker,
    		  Trajectory* trackerTraj) :
      theTrajectory(traj), theMuonTrack(muon), theTrackerTrack(tracker), theTrackerTrajectory(trackerTraj) {} 
    
    MuonCandidate(Trajectory* traj, 
                  const reco::TrackRef& muon, 
                  const reco::TrackRef& tracker) :
      theTrajectory(traj), theMuonTrack(muon), theTrackerTrack(tracker), theTrackerTrajectory(0) {} 
    
    /// destructor
    virtual ~MuonCandidate() { }
  
    /// return trajectory
    Trajectory* trajectory() const { return theTrajectory; }

    /// return muon track
    const reco::TrackRef muonTrack() const { return theMuonTrack; }

    /// return tracker track
    const reco::TrackRef trackerTrack() const { return theTrackerTrack; }
    
    /// return tracker trajectory
    Trajectory* trackerTrajectory() const { return theTrackerTrajectory; }
                 
  private:

    Trajectory* theTrajectory;
    reco::TrackRef theMuonTrack;
    reco::TrackRef theTrackerTrack;
    Trajectory* theTrackerTrajectory;

};
#endif 
