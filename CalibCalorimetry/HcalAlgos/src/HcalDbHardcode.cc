
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbHardcode.cc,v 1.9 2006/05/26 23:32:03 fedor Exp $
//
#include <vector>
#include <string>

#include "CLHEP/Random/RandGauss.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbHardcode.h"


HcalPedestal HcalDbHardcode::makePedestal (HcalDetId fId, bool fSmear) {
  HcalGain gain = HcalDbHardcode::makeGain (fId, fSmear);
  HcalPedestalWidth width = makePedestalWidth (fId);
  float value0 = 0.75;
  float value [4] = {value0, value0, value0, value0};
  if (fSmear) {
    for (int i = 0; i < 4; i++) {
      value [i] = RandGauss::shoot (value0, width.getWidth (i) / 100.); // ignore correlations, assume 10K pedestal run 
      while (value [i] <= 0) value [i] = RandGauss::shoot (value0, width.getWidth (i));
    }
  }
  HcalPedestal result (fId.rawId (), 
		       value[0] / gain.getValue (0),
		       value[1] / gain.getValue (1),
		       value[2] / gain.getValue (2),
		       value[3] / gain.getValue (3)
		       );
  return result;
}

HcalPedestalWidth HcalDbHardcode::makePedestalWidth (HcalDetId fId) {
  HcalGain gain = HcalDbHardcode::makeGain (fId);
  float value = fId.subdet () == HcalForward ? 0.14 : 0.1;
  HcalPedestalWidth result (fId.rawId ());
  for (int i = 0; i < 4; i++) {
    double width = value / gain.getValue (i);
    for (int j = 0; j <= i; j++) {
      result.setSigma (i, j, i == j ? width * width : 0);
    }
  } 
  return result;
}

HcalGain HcalDbHardcode::makeGain (HcalDetId fId, bool fSmear) {
  HcalGainWidth width = makeGainWidth (fId);
  float value0 = fId.subdet () == HcalForward ? 0.48 : 0.177; // GeV/fC
  float value [4] = {value0, value0, value0, value0};
  if (fSmear) for (int i = 0; i < 4; i++) value [i] = RandGauss::shoot (value0, width.getValue (i)); 
  HcalGain result (fId.rawId (), value[0], value[1], value[2], value[3]);
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

HcalCalibrationQIECoder HcalDbHardcode::makeCalibrationQIECoder (HcalDetId fId) {
  HcalCalibrationQIECoder result (fId.rawId ());
  float lowEdges [32];
  for (int i = 0; i < 32; i++) lowEdges[i] = -1.5 + i*0.35;
  result.setMinCharges (lowEdges);
  return result;
}

HcalQIEShape HcalDbHardcode::makeQIEShape () {
  return HcalQIEShape ();
}

