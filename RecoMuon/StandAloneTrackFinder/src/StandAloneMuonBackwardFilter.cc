/** \class StandAloneMuonBackwardFilter
 *  The outward-inward fitter (starts from StandAloneMuonFilter outermost state).
 *
 *  $Date: 2006/08/31 18:28:04 $
 *  $Revision: 1.5 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonBackwardFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

StandAloneMuonBackwardFilter::StandAloneMuonBackwardFilter(const ParameterSet& par,const MuonServiceProxy* service){

}

void StandAloneMuonBackwardFilter::setEvent(const Event& event){}
