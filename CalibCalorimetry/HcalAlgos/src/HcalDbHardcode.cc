
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbASCIIIO.cc,v 1.2 2005/12/05 00:25:30 fedor Exp $
//
#include <vector>
#include <string>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"


HcalPedestal HcalDbHardcode::makePedestal (HcalDetId fId) {
  HcalGain gain = HcalDbHardcode::makeGain (fId);
  float value = 0.75;
  HcalPedestal result (fId.rawId (), 
		       value / gain.getValue (1),
		       value / gain.getValue (2),
		       value / gain.getValue (3),
		       value / gain.getValue (4)
		       );
  return result;
}

HcalPedestalWidth HcalDbHardcode::makePedestalWidth (HcalDetId fId) {
  float value = 0;
  HcalPedestalWidth result (fId.rawId (), value, value, value, value);
  return result;
}

HcalGain HcalDbHardcode::makeGain (HcalDetId fId) {
  float value = fId.subdet () == HcalForward ? 0.150 : 0.177; // GeV/fC
  HcalGain result (fId.rawId (), value, value, value, value);
  return result;
}

HcalGainWidth HcalDbHardcode::makeGainWidth (HcalDetId fId) {
  float value = 0;
  HcalGainWidth result (fId.rawId (), value, value, value, value);
  return result;
}

HcalQIECoder HcalDbHardcode::makeQIECoder (HcalDetId fId) {
  HcalQIECoder result (fId.rawId ());
  float offset = 0;
  float slope = fId.subdet () == HcalForward ? 2.6 : 1.;
  for (unsigned range = 0; range < 4; range++) {
    for (unsigned capid = 0; capid < 4; capid++) {
      result.setOffset (capid, range, offset);
      result.setSlope (capid, range, slope);
    }
  }
  return result;
}

HcalQIEShape HcalDbHardcode::makeQIEShape () {
  return HcalQIEShape ();
}

