/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/03/02 19:47:32 $
 *  $Revision: 1.3 $
 *  \author A. Vilela Pereira
 */

#include "DTTTrigMatchRPhi.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <math.h>

using namespace std;
using namespace edm;

namespace dtCalibration {

DTTTrigMatchRPhi::DTTTrigMatchRPhi(const ParameterSet& pset) {
  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");
}

DTTTrigMatchRPhi::~DTTTrigMatchRPhi() {}

void DTTTrigMatchRPhi::setES(const EventSetup& setup) {
  // Get tTrig record from DB
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
  tTrigMap_ = &*tTrig;
}

DTTTrigData DTTTrigMatchRPhi::correction(const DTSuperLayerId& slId) {
  
  float tTrigMean,tTrigSigma,kFactor;
  int status = tTrigMap_->get(slId,tTrigMean,tTrigSigma,kFactor,DTTimeUnits::ns);
  // RZ superlayers return the current value
  if(slId.superLayer() == 2){
    if(status != 0) throw cms::Exception("[DTTTrigMatchRPhi]") << "Could not find tTrig entry in DB for"
                                                               << slId << endl;
    return DTTTrigData(tTrigMean,tTrigSigma,kFactor);
  } else{
    DTSuperLayerId partnerSLId(slId.chamberId(),(slId.superLayer() == 1)?3:1);
    float tTrigMeanNew,tTrigSigmaNew,kFactorNew;
    if(!status){ // Gets average of both SuperLayer's
      if(!tTrigMap_->get(partnerSLId,tTrigMeanNew,tTrigSigmaNew,kFactorNew,DTTimeUnits::ns)){
        tTrigMeanNew = (tTrigMean + tTrigMeanNew)/2.;
//         tTrigSigmaNew = sqrt(tTrigSigmaNew*tTrigSigmaNew + tTrigSigma*tTrigSigma)/2.;
        tTrigSigmaNew = (tTrigSigmaNew + tTrigSigma)/2.;

	kFactorNew = kFactor;
        return DTTTrigData(tTrigMeanNew,tTrigSigmaNew,kFactorNew);
      } else return DTTTrigData(tTrigMean,tTrigSigma,kFactor); 
    } else{ // If there is no entry tries to find partner SL and retrieves its value
      if(!tTrigMap_->get(partnerSLId,tTrigMeanNew,tTrigSigmaNew,kFactorNew,DTTimeUnits::ns))
	return DTTTrigData(tTrigMeanNew,tTrigSigmaNew,kFactorNew);
      else { // Both RPhi SL's not present in DB
        throw cms::Exception("[DTTTrigMatchRPhi]") << "Could not find tTrig entry in DB for"
                                                   << slId << "\n" << partnerSLId << endl;
      }
    }
  }
}

} // namespace
