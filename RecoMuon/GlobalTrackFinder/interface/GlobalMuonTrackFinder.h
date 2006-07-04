#ifndef GlobalTrackFinder_GlobalMuonTrackFinder_H
#define GlobalTrackFinder_GlobalMuonTrackFinder_H

/** \class GlobalMuonTrackFinder
 *  class to build (Tk and Combined) Tracks from standalone muon Track
 *  using GlobalMuonTrajectoryBuilder
 *
 *  $Date: 2006/07/02 03:00:56 $
 *  $Revision: 1.2 $
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

class GlobalMuonTrackFinder{

public:

  /// constructor
  GlobalMuonTrackFinder(GlobalMuonTrajectoryBuilder*);
          
  /// destructor 
  ~GlobalMuonTrackFinder();

  /// reconstruct muon and put into Event 
  void reconstruct(const edm::Handle<reco::TrackCollection>&,
		   edm::Event&,
		   const edm::EventSetup&) const;

private:

    std::string theTkTrackLabel;
    GlobalMuonTrajectoryBuilder*  theTrajectoryBuilder;

};
#endif

