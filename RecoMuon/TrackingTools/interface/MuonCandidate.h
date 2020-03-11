#ifndef RecoMuon_TrackingTools_MuonCandidate_H
#define RecoMuon_TrackingTools_MuonCandidate_H

/** \class MuonCandidate
 *  Auxiliary class for muon candidates
 *
 *  \author N. Neumeister	Purdue University 
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>
#include <memory>

class MuonCandidate {
public:
  typedef std::vector<std::unique_ptr<Trajectory>> TrajectoryContainer;
  typedef std::vector<std::unique_ptr<MuonCandidate>> CandidateContainer;

public:
  /// constructor
  MuonCandidate(std::unique_ptr<Trajectory> traj,
                const reco::TrackRef& muon,
                const reco::TrackRef& tracker,
                std::unique_ptr<Trajectory> trackerTraj)
      : theTrajectory(std::move(traj)),
        theMuonTrack(muon),
        theTrackerTrack(tracker),
        theTrackerTrajectory(std::move(trackerTraj)) {}

  MuonCandidate(std::unique_ptr<Trajectory> traj, const reco::TrackRef& muon, const reco::TrackRef& tracker)
      : theTrajectory(std::move(traj)), theMuonTrack(muon), theTrackerTrack(tracker), theTrackerTrajectory(nullptr) {}

  /// destructor
  virtual ~MuonCandidate() {}

  /// return trajectory
  Trajectory const* trajectory() const { return theTrajectory.get(); }

  std::unique_ptr<Trajectory> releaseTrajectory() { return std::move(theTrajectory); }

  /// return muon track
  const reco::TrackRef muonTrack() const { return theMuonTrack; }

  /// return tracker track
  const reco::TrackRef trackerTrack() const { return theTrackerTrack; }

  /// return tracker trajectory
  Trajectory const* trackerTrajectory() const { return theTrackerTrajectory.get(); }
  std::unique_ptr<Trajectory> releaseTrackerTrajectory() { return std::move(theTrackerTrajectory); }

private:
  std::unique_ptr<Trajectory> theTrajectory;
  reco::TrackRef theMuonTrack;
  reco::TrackRef theTrackerTrack;
  std::unique_ptr<Trajectory> theTrackerTrajectory;
};
#endif
