///////////////////////////////////////////////////////////////////////////////
// File: HBDarkening.cc
// Description: simple helper class containing parameterized function 
//              to be used for the SLHC darkening calculation in HB 
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/HcalCalibObjects/interface/HBDarkening.h"
#include <cmath>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DebugLog

HBDarkening::HBDarkening(unsigned int scenario) {
  //HB starts at tower 1
  ieta_shift = 1;
  
  //scale parameter in fb-1 for exponential darkening from radiation damage for each tile
  //extrapolated from HE radiation damage model (HEDarkening) using FLUKA dose map
  //notes: extrapolation ignores geometrical and material differences in HB vs. HE
  //       which affects the material averaging in the coarse FLUKA dose map
  //       also, atmosphere in the barrel has more N2 and less O2 vs. HE
  //conclusion: very rough extrapolation, caveat emptor!
  float _lumiscale[nEtaBins][nScintLayers] = {
      {887.1,887.1,1199.3,1314.0,1615.8,1954.0,2504.1,2561.7,3249.6,4553.7,5626.1,5725.6,6777.4,8269.8,10061.8,15253.3,22200.9},
      {990.8,990.8,1140.8,1325.4,1676.9,2036.6,2900.5,2789.1,3460.2,4405.9,4184.0,5794.6,6157.4,7646.0,11116.1,18413.9,26813.3},
      {971.2,971.2,1244.3,1456.2,1760.1,2299.1,2603.2,3012.3,3933.9,4787.3,4503.9,6624.3,7059.4,9369.5,12038.0,20048.8,23541.3},
      {877.3,877.3,1145.2,1322.5,1604.9,1924.0,2893.6,2827.4,4085.0,5320.2,4740.6,5693.5,5715.1,7373.4,8305.1,16079.9,21702.3},
      {919.8,919.8,1223.2,1376.9,1742.6,1964.7,2494.7,3335.6,4520.6,4869.5,4895.6,5740.4,7089.8,8765.9,10045.7,17408.0,24726.7},
      {901.1,901.1,1114.3,1391.2,1733.2,2210.7,2733.8,3399.4,3715.2,3626.3,3371.3,4653.8,5911.6,7204.2,7584.7,4760.1,11156.8},
      {904.0,904.0,1112.9,1351.9,1722.3,2008.2,2709.8,3101.9,3470.5,4679.0,5843.6,6343.8,7883.3,11266.8,16607.2,10882.3,25428.4},
      {930.7,930.7,1225.3,1341.9,1744.0,2253.8,2805.1,3329.9,3665.6,5179.6,5677.8,5753.0,5662.3,9516.5,10769.4,13892.9,16661.1},
      {953.2,953.2,1240.4,1487.3,1719.8,1930.6,2595.7,3172.5,3881.0,5247.5,4934.0,6576.4,6353.7,9259.2,12264.5,13261.8,12222.6},
      {877.4,877.4,1114.1,1346.0,1604.9,1997.6,2708.9,3247.9,3704.4,4568.2,4984.4,7000.8,7896.7,7970.0,12555.6,10062.1,18386.8},
      {876.2,876.2,1127.1,1336.0,1753.1,1944.4,2641.1,3445.1,3810.2,4033.0,5301.2,5170.7,6062.3,9815.8,12854.2,14297.3,20692.1},
      {841.2,841.2,1051.1,1229.5,1576.5,1983.2,2282.2,2981.2,3271.7,4417.1,3765.2,4491.8,4626.6,7173.2,12953.0,7861.2,19338.6},
      {848.2,848.2,1072.8,1228.5,1497.9,1876.1,2279.7,2744.2,3325.9,4021.8,4081.1,3750.6,4584.2,6170.8,9020.5,12058.8,13492.2},
      {780.3,780.3,977.6,1103.2,1323.2,1548.3,1970.1,2217.5,2761.0,3049.0,2913.0,3832.3,4268.6,5242.1,6432.8,5999.5,6973.9},
      {515.2,515.2,662.1,836.5,1110.4,1214.4,1664.4,1919.0,2341.0,2405.2,2647.5,2593.8,2586.3,2814.4,2826.8,0.0,0.0},
      {409.3,409.3,489.2,700.3,960.2,1103.5,909.8,934.6,1148.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}
  };

  //store array
  //no factor needed to account for increased flux at 14 TeV vs 7-8 TeV
  //already included in extrapolation from HE
  for(unsigned int j = 0; j < nEtaBins; j++){
    for(unsigned int i = 0; i < nScintLayers; i++){
	  if (scenario == 0) lumiscale[j][i] = 0;
      else lumiscale[j][i] = _lumiscale[j][i];
	}
  } 
}

HBDarkening::~HBDarkening() { }

float HBDarkening::degradation(float intlumi, int ieta, int lay) const {
  //no lumi, no darkening
  if(intlumi <= 0) return 1.;

  //shift ieta tower index
  ieta -= ieta_shift;
  
  //if outside eta range, no darkening
  if(ieta < 0 || ieta >= (int)nEtaBins) return 1.;
  
  //layer index does not need a shift in HB
  
  //if outside layer range, no darkening
  if(lay < 0 || lay >= (int)nScintLayers) return 1.;

  // if replaced tile by scenario
  if (lumiscale[ieta][lay] == 0.0) return 1.;

  //return darkening factor
  return (exp(-intlumi/lumiscale[ieta][lay]));
}

const char* HBDarkening::scenarioDescription (unsigned int scenario) {
  if (scenario == 0) return "full replacement of HE scintillators, no darkening";
  else if (scenario == 1) return "no replacement, full stage darkening";
  return "undefined scenario: assume no replacement, full stage darkening";
}
