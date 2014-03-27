///////////////////////////////////////////////////////////////////////////////
// File: HEDarkening.cc
// Description: simple helper class containing parameterized function 
//              to be used for the SLHC darkening calculation in HE 
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/HcalCalibObjects/interface/HEDarkening.h"
#include <cmath>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DebugLog

HEDarkening::HEDarkening() {
  //HE starts at tower 16
  ieta_shift = 16;
  
  //scale parameter in fb-1 for exponential darkening from radiation damage for each tile
  //fits from laser data for L1 and L7
  //L0,Lm1 = L1, L2-L6 interpolated, L8-L17 extrapolated
  //from Vladimir Epshteyn
  float _lumiscale[nEtaBins][nScintLayers] = {
      {1194.9, 1194.9, 1194.9, 1651.5, 2282.7, 3155.2, 4361.0, 6027.8, 8331.5, 11515.7, 15916.8, 22000.0, 30408.2, 42029.8, 58093.1, 80295.6, 110983.5, 153400.1, 212027.7},
      {952.8, 952.8, 952.8, 1293.9, 1757.1, 2386.1, 3240.3, 4400.3, 5975.4, 8114.5, 11019.3, 14963.9, 20320.6, 27594.9, 37473.2, 50887.7, 69104.3, 93841.9, 127435.0},
      {759.8, 759.8, 759.8, 1013.8, 1352.5, 1804.5, 2407.6, 3212.2, 4285.7, 5717.9, 7628.7, 10178.1, 13579.5, 18117.6, 24172.3, 32250.4, 43028.0, 57407.4, 76592.2},
      {605.9, 605.9, 605.9, 794.2, 1041.1, 1364.7, 1788.9, 2344.9, 3073.7, 4029.1, 5281.4, 6922.9, 9074.7, 11895.2, 15592.4, 20438.8, 26791.5, 35118.8, 46034.2},
      {483.2, 483.2, 483.2, 622.3, 801.4, 1032.1, 1329.2, 1711.8, 2204.5, 2839.1, 3656.3, 4708.8, 6064.3, 7809.9, 10058.0, 12953.2, 16681.8, 21483.8, 27667.9},
      {385.3, 385.3, 385.3, 487.5, 616.9, 780.5, 987.6, 1249.6, 1581.1, 2000.6, 2531.3, 3202.8, 4052.5, 5127.6, 6487.9, 8209.2, 10387.0, 13142.6, 16629.3},
      {307.3, 307.3, 307.3, 382.0, 474.8, 590.3, 733.8, 912.2, 1134.0, 1409.7, 1752.4, 2178.5, 2708.1, 3366.6, 4185.1, 5202.6, 6467.5, 8039.9, 9994.7},
      {245.0, 245.0, 245.0, 299.3, 365.5, 446.4, 545.2, 665.9, 813.3, 993.3, 1213.2, 1481.8, 1809.7, 2210.3, 2699.6, 3297.2, 4027.0, 4918.4, 6007.1},
      {195.4, 195.4, 195.4, 234.5, 281.3, 337.6, 405.1, 486.1, 583.3, 700.0, 839.9, 1007.9, 1209.4, 1451.2, 1741.4, 2089.6, 2507.4, 3008.8, 3610.5},
      {155.8, 155.8, 155.8, 183.7, 216.6, 255.3, 301.0, 354.9, 418.4, 493.2, 581.5, 685.5, 808.2, 952.8, 1123.3, 1324.3, 1561.3, 1840.6, 2170.0},
      {124.3, 124.3, 124.3, 143.9, 166.7, 193.1, 223.6, 259.0, 300.1, 347.5, 402.6, 466.3, 540.1, 625.6, 724.6, 839.3, 972.1, 1126.0, 1304.2},
      {99.1, 99.1, 99.1, 112.8, 128.3, 146.0, 166.2, 189.1, 215.2, 244.9, 278.7, 317.2, 360.9, 410.7, 467.4, 531.9, 605.3, 688.8, 783.9},
      {79.0, 79.0, 79.0, 88.3, 98.8, 110.4, 123.5, 138.0, 154.3, 172.6, 192.9, 215.7, 241.2, 269.7, 301.5, 337.1, 376.9, 421.4, 471.1},
      {63.0, 63.0, 63.0, 69.2, 76.0, 83.5, 91.7, 100.8, 110.7, 121.6, 133.6, 146.7, 161.2, 177.0, 194.5, 213.6, 234.7, 257.8, 283.2}
	};

  //store array
  //flux_factor: to account for increased flux at 14 TeV vs 7-8 TeV (approximate)
  //*divide* lumiscale params by this since increased flux -> faster darkening
  double flux_factor = 1.2;
  for(unsigned int j = 0; j < nEtaBins; j++){
    for(unsigned int i = 0; i < nScintLayers; i++){
	  lumiscale[j][i] = _lumiscale[j][i]/flux_factor;
	}
  } 

}

HEDarkening::~HEDarkening() { }

float HEDarkening::degradation(float intlumi, int ieta, int lay) {
  //no lumi, no darkening
  if(intlumi <= 0) return 1.;

  //shift ieta tower index
  ieta -= ieta_shift;
  
  //if outside eta range, no darkening
  if(ieta < 0 || ieta >= (int)nEtaBins) return 1.;
  
  //shift layer index by 1 to act as array index
  lay += 1;
  
  //if outside layer range, no darkening
  if(lay < 0 || lay >= (int)nScintLayers) return 1.;

  //return darkening factor
  return (exp(-intlumi/lumiscale[ieta][lay]));
}
