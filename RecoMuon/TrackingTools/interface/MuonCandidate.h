#ifndef MuonCandidate_h
#define MuonCandidate_h

/** \class MuonCandidate
 *
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class MuonCandidate {
 public:

  typedef std::vector<Trajectory* > TrajectoryContainer;
  typedef std::vector<MuonCandidate* > CandidateContainer;

  /// Constructor
  MuonCandidate();

  /// Destructor
  virtual ~MuonCandidate();

  const Trajectory& trajectory() {return theTrajectory;}
  const reco::TrackRef trackerTrack() {return theTrackerTrack;}
  const reco::TrackRef muonTrack() {return theMuonTrack;}

 private:
  Trajectory theTrajectory;
  reco::TrackRef theTrackerTrack;
  reco::TrackRef theMuonTrack;

};
#endif

