/** \class StandAloneMuonBackwardFilter
 *  The outward-inward fitter (starts from StandAloneMuonRefitter outermost state).
 *
 *  $Date: 2006/08/30 12:56:19 $
 *  $Revision: 1.4 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonBackwardFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

StandAloneMuonBackwardFilter::StandAloneMuonBackwardFilter(const ParameterSet& par,const MuonServiceProxy* service){

}

void StandAloneMuonBackwardFilter::setEvent(const Event& event){}
