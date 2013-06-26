/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/03/21 13:48:06 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "DTTTrigConstantShift.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <math.h>

using namespace std;
using namespace edm;

namespace dtCalibration {

DTTTrigConstantShift::DTTTrigConstantShift(const ParameterSet& pset):
  dbLabel_( pset.getUntrackedParameter<string>("dbLabel", "") ),
  calibChamber_( pset.getParameter<string>("calibChamber") ),
  value_( pset.getParameter<double>("value") ) {

  LogVerbatim("Calibration") << "[DTTTrigConstantShift] Applying constant correction value: " << value_ << endl;

  if( calibChamber_ != "" && calibChamber_ != "None" && calibChamber_ != "All" ){
    stringstream linestr;
    int selWheel, selStation, selSector;
    linestr << calibChamber_;
    linestr >> selWheel >> selStation >> selSector;
    chosenChamberId_ = DTChamberId(selWheel, selStation, selSector);
    LogVerbatim("Calibration") << "[DTTTrigConstantShift] Chosen chamber: " << chosenChamberId_ << endl;
  }
  //FIXME: Check if chosen chamber is valid.
}

DTTTrigConstantShift::~DTTTrigConstantShift() {}

void DTTTrigConstantShift::setES(const EventSetup& setup) {
  // Get tTrig record from DB
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(dbLabel_,tTrig);
  tTrigMap_ = &*tTrig;
}

DTTTrigData DTTTrigConstantShift::correction(const DTSuperLayerId& slId) {
  
  float tTrigMean,tTrigSigma,kFactor;
  int status = tTrigMap_->get(slId,tTrigMean,tTrigSigma,kFactor,DTTimeUnits::ns);
  if(status != 0) throw cms::Exception("[DTTTrigConstantShift]") << "Could not find tTrig entry in DB for"
                                                                 << slId << endl;

  float tTrigMeanNew = tTrigMean;
  if( calibChamber_ != "" && calibChamber_ != "None"){
     if( ( calibChamber_ == "All" ) ||
         ( calibChamber_ != "All" && slId.chamberId() == chosenChamberId_ ) ) {
        tTrigMeanNew = tTrigMean + value_; 
     }
  }

  return DTTTrigData(tTrigMeanNew,tTrigSigma,kFactor);
}

} // namespace
