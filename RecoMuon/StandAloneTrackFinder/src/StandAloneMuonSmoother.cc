/** \class StandAloneMuonSmoother
 *  The outward-inward fitter (starts from StandAloneMuonBackwardFilter innermost state).
 *
 *  $Date: 2006/08/30 12:56:19 $
 *  $Revision: 1.4 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// #include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonSmoother.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

StandAloneMuonSmoother::StandAloneMuonSmoother(const ParameterSet& par, const MuonServiceProxy* service){}

void StandAloneMuonSmoother::setEvent(const Event& event){}
