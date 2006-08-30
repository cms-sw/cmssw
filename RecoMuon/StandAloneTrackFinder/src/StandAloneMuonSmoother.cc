/** \class StandAloneMuonSmoother
 *  The outward-inward fitter (starts from StandAloneMuonBackwardFilter innermost state).
 *
 *  $Date: 2006/05/23 17:47:24 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// #include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonSmoother.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

StandAloneMuonSmoother::StandAloneMuonSmoother(const ParameterSet& par, const MuonServiceProxy* service){}

void StandAloneMuonSmoother::setES(const EventSetup& setup){}
void StandAloneMuonSmoother::setEvent(const Event& event){}
