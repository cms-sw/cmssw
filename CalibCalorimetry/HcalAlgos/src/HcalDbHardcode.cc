
//
// F.Ratnikov (UMd), Dec 14, 2005
// $Id: HcalDbHardcode.cc,v 1.10 2006/09/11 20:29:08 fedor Exp $
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
  if (fId.subdet () != HcalForward) {
    // replicate channel eta=18, phi=17, depth=2
    result.setOffset (0, 0, -0.2275);  result.setSlope (0, 0, 0.8957);
    result.setOffset (0, 1,  4.2746);  result.setSlope (0, 1, 0.8901);
    result.setOffset (0, 2,  5.6746);  result.setSlope (0, 2, 0.8905);
    result.setOffset (0, 3, 54.6185);  result.setSlope (0, 3, 0.8926);
    result.setOffset (1, 0, -0.4920);  result.setSlope (1, 0, 0.9038);
    result.setOffset (1, 1,  4.0778);  result.setSlope (1, 1, 0.9040);
    result.setOffset (1, 2,  7.4032);  result.setSlope (1, 2, 0.9001);
    result.setOffset (1, 3, 47.7975);  result.setSlope (1, 3, 0.9044);
    result.setOffset (2, 0, -0.5608);  result.setSlope (2, 0, 0.8833);
    result.setOffset (2, 1,  3.4970);  result.setSlope (2, 1, 0.8742);
    result.setOffset (2, 2,  9.0395);  result.setSlope (2, 2, 0.8702);
    result.setOffset (2, 3, 12.4871);  result.setSlope (2, 3, 0.8654);
    result.setOffset (3, 0, -0.1053);  result.setSlope (3, 0, 0.9095);
    result.setOffset (3, 1,  4.8267);  result.setSlope (3, 1, 0.9086);
    result.setOffset (3, 2, 11.0497);  result.setSlope (3, 2, 0.9115);
    result.setOffset (3, 3, 49.0923);  result.setSlope (3, 3, 0.8945);
   }
  else {
    // replicate channel eta=30, phi=1, depth=1
    result.setOffset (0, 0, -0.4083);  result.setSlope (0, 0, 0.3553);
    result.setOffset (0, 1,  2.2721);  result.setSlope (0, 1, 0.3547);
    result.setOffset (0, 2,  9.5776);  result.setSlope (0, 2, 0.3531);
    result.setOffset (0, 3,-76.3834);  result.setSlope (0, 3, 0.3529);
    result.setOffset (1, 0, -0.4720);  result.setSlope (1, 0, 0.3556);
    result.setOffset (1, 1,  3.2788);  result.setSlope (1, 1, 0.3545);
    result.setOffset (1, 2,  4.7199);  result.setSlope (1, 2, 0.3545);
    result.setOffset (1, 3,-49.8731);  result.setSlope (1, 3, 0.3538);
    result.setOffset (2, 0, -0.5847);  result.setSlope (2, 0, 0.3484);
    result.setOffset (2, 1,  2.0515);  result.setSlope (2, 1, 0.3496);
    result.setOffset (2, 2,  1.6429);  result.setSlope (2, 2, 0.3502);
    result.setOffset (2, 3, -42.734);  result.setSlope (2, 3, 0.3501);
    result.setOffset (3, 0, -0.6760);  result.setSlope (3, 0, 0.3688);
    result.setOffset (3, 1,  3.5937);  result.setSlope (3, 1, 0.3667);
    result.setOffset (3, 2,  0.1241);  result.setSlope (3, 2, 0.3649);
    result.setOffset (3, 3,-58.8199);  result.setSlope (3, 3, 0.3641);
  }
  return result;
}

HcalCalibrationQIECoder HcalDbHardcode::makeCalibrationQIECoder (HcalDetId fId) {
  HcalCalibrationQIECoder result (fId.rawId ());
  float lowEdges [32];
  if (fId.subdet () != HcalForward) {
    // replicate channel eta=18, phi=17, depth=2
    float lowEdgesHbHeHo [32] = {-1.91050, -1.56210, -1.21380, -0.86550, -0.51710,
				 -0.16880,  0.17960,  0.52790,  0.87240,  1.22230,
				 1.57110,  1.92230,  2.27130,  2.62150,  2.97080,
				 3.31770,  3.66480,  4.01270,  4.36290,  4.71030,
				 5.05430,  5.40420,  5.75060,  6.09530,  6.44980,
				 6.79820,  7.14650,  7.49490,  7.84320,  8.19160,
				 8.53990,  8.88830};
    for (int i = 0; i < 32; i++) lowEdges[i] = lowEdgesHbHeHo [i];
  }
  else {
    // replicate channel eta=30, phi=1, depth=1
    float lowEdgesHf [32] = {-6.76920, -5.83720, -4.90510, -3.97310, -3.04110,
			     -2.10900, -1.17700, -0.24500,  0.68710,  1.61910,
			     2.56190,  3.48640,  4.41140,  5.33960,  6.27050, 
			     7.20570,  8.14000,  9.07160, 10.00250, 10.93380,
			     11.86920, 12.80830, 13.74580, 14.68020, 15.61550,
			     16.55230, 17.46380, 17.78070, 19.32790, 20.25990,
			     21.19190, 22.12400};
    for (int i = 0; i < 32; i++) lowEdges[i] = lowEdgesHf [i];
  }
  result.setMinCharges (lowEdges);
  return result;
}

HcalQIEShape HcalDbHardcode::makeQIEShape () {
  return HcalQIEShape ();
}

