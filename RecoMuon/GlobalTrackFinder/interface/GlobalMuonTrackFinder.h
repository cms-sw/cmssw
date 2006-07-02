#ifndef GlobalTrackFinder_GlobalMuonTrackFinder_H
#define GlobalTrackFinder_GlobalMuonTrackFinder_H

/** \class GlobalMuonTrackFinder
 *  class to build (Tk and Combined) Tracks from standalone muon Track
 *  using GlobalMuonTrajectoryBuilder
 *
 *  $Date: 2006/06/30 03:31:11 $
 *  $Revision: 1.1 $
 *  \author C. Liu - Purdue University
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>

namespace edm {class ParameterSet;}

class GlobalMuonTrajectoryBuilder;
class GlobalMuonTrackMatcher;

class GlobalMuonTrackFinder{

public:

  /// constructor
  GlobalMuonTrackFinder(GlobalMuonTrajectoryBuilder*,GlobalMuonTrackMatcher*);
          
  /// destructor 
  ~GlobalMuonTrackFinder();

  /// reconstruct muon and put into Event 
  void reconstruct(const edm::Handle<reco::TrackCollection>&,
		   edm::Event&,
		   const edm::EventSetup&) const;

private:

    GlobalMuonTrajectoryBuilder*  theTrajectoryBuilder;
    GlobalMuonTrackMatcher* theTrackMatcher;

};
#endif

