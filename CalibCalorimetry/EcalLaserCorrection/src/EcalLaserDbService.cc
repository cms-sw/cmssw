
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

//#include "RecoLocalCalo/CaloTowersCreator/interface/EScales.h"


EcalLaserDbService::EcalLaserDbService () 
  : 
  mAlphas (0),
  mAPDPNRatiosRef (0),
  mAPDPNRatios (0)
 {}


const EcalLaserAlphas* EcalLaserDbService::getAlphas () const {
  return mAlphas;
}

const EcalLaserAPDPNRatiosRef* EcalLaserDbService::getAPDPNRatiosRef () const {
  return mAPDPNRatiosRef;
}

const EcalLaserAPDPNRatios* EcalLaserDbService::getAPDPNRatios () const {
  return mAPDPNRatios;
}


//const HcalElectronicsMap* EcalLaserDbService::getHcalMapping () const {
//  return mElectronicsMap;
//}

EVENTSETUP_DATA_REG(EcalLaserDbService);
