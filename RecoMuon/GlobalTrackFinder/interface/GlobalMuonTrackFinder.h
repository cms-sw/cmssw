#ifndef GlobalTrackFinder_GlobalMuonTrackFinder_H
#define GlobalTrackFinder_GlobalMuonTrackFinder_H

/** \class GlobalMuonTrackFinder
 *  class to build (Tk and Combined) Tracks from standalone muon Track
 *  using GlobalMuonTrajectoryBuilder
 *
 *  $Date:  $
 *  $Revision: $
 *  \author C. Liu - Purdue University
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>

namespace edm {class ParameterSet;}

class GlobalMuonTrackFinder{

public:

  /// constructor
  GlobalMuonTrackFinder(const GlobalMuonTrajectoryBuilder *);
          
  /// destructor 
  ~GlobalMuonTrackFinder();

  /// reconstruct muon and put into Event 
  void reconstruct(const edm::Handle<TrackCollection>&,
		   edm::Event&,
		   const edm::EventSetup&);

private:

    GlobalMuonTrajectoryBuilder*  theTrajectoryBuilder;

};
#endif

