/** \class StandAloneMuonBackwardFilter
 *  The outward-inward fitter (starts from StandAloneMuonRefitter outermost state).
 *
 *  $Date: 2006/05/19 15:24:36 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonBackwardFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

StandAloneMuonBackwardFilter::StandAloneMuonBackwardFilter(const ParameterSet& par){

}

void StandAloneMuonBackwardFilter::setES(const EventSetup& setup){}
void StandAloneMuonBackwardFilter::setEvent(const Event& event){}
