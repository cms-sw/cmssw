/** \class StandAloneMuonRefitter
 *  Class ti interface the muon system rechits with the standard KF tools.
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par, const MuonServiceProxy* service):theService(service){
  
}

/// Destructor
StandAloneMuonRefitter::~StandAloneMuonRefitter(){

}

  // Operations

  /// Refit
StandAloneMuonRefitter::RefitResult StandAloneMuonRefitter::refit(const Trajectory&){


}
