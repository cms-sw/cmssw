#ifndef GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H
#define GlobalTrackFinder_GlobalMuonTrajectoryBuilder_H

/** \class GlobalMuonTrajectoryBuilder
 *  class to build muon trajectory
 *
 *  $Date: 2006/07/11 03:40:39 $
 *  $Revision: 1.6 $
 *  \author Chang Liu - Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/EventSetup.h"
//#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace edm {class ParameterSet;}

class GlobalMuonTrajectoryBuilder : public MuonTrajectoryBuilder{

public:

  typedef edm::OwnVector< const TransientTrackingRecHit>  RecHitContainer;
  //typedef std::vector<Trajectory> TrajectoryContainer;
  //typedef std::pair<Trajectory, reco::TrackRef&> MuonCandidate;
  //typedef std::vector<MuonCandidate> CandidateContainer;
  //typedef TrajectoryContainer::const_iterator TCI;
 
  /** Constructor with Parameter Set */
  GlobalMuonTrajectoryBuilder(const edm::ParameterSet& par);
          
  /** Destructor */
  ~GlobalMuonTrajectoryBuilder();

  /** Returns a vector of the reconstructed trajectories compatible with
   * the given seed.
   */
  CandidateContainer trajectories(const reco::Track&);
  TrajectoryContainer trajectories(const TrajectorySeed&) {TrajectoryContainer result; return result;}
  
 private:
   
  // Pass the Event Setup to the algo at each event
  virtual void setES(const edm::EventSetup& setup);
  
  /// Pass the Event to the algo at each event
  virtual void setEvent(const edm::Event& event);
  
   
};
#endif
