/** \class StandAloneMuonSmoother
 *  The outward-inward fitter (starts from StandAloneMuonBackwardFilter innermost state).
 *
 *  $Date: 2006/05/18 09:53:21 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

// #include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonSmoother.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

StandAloneMuonSmoother::StandAloneMuonSmoother(const ParameterSet& par){}

void StandAloneMuonSmoother::setES(const EventSetup& setup){}
