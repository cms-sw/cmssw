/** \class StandAloneMuonBackwardFilter
 *  The outward-inward fitter (starts from StandAloneMuonFilter outermost state).
 *
 *  $Date: 2008/04/23 16:56:34 $
 *  $Revision: 1.6 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

//#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonBackwardFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace edm;

StandAloneMuonBackwardFilter::StandAloneMuonBackwardFilter(const ParameterSet& par,const MuonServiceProxy* service){

}

void StandAloneMuonBackwardFilter::setEvent(const Event& event){}
