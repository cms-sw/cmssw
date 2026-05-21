#ifndef HeavyIonsAnalysis_ZDCAnalysis_plugins_ZDCHardCodeHelper
#define HeavyIonsAnalysis_ZDCAnalysis_plugins_ZDCHardCodeHelper

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include <vector>

using namespace std;
class ZDCHardCodeHelper {
public:
  ZDCHardCodeHelper();
  virtual ~ZDCHardCodeHelper();
  // int  eta2ieta(double eta);
  double charge(const QIE10DataFrame& digi, int ts);
  double charge( unsigned fAdc, unsigned fCapId);

  int rechit_Energy_TriggerBit_EM(const QIE10DataFrame& digi);
  int rechit_Energy_TriggerBit_HAD(const QIE10DataFrame& digi);
  
  double rechit_Energy_Trigger_EM(const QIE10DataFrame& digi);
  double rechit_Energy_Trigger_HAD(const QIE10DataFrame& digi);
  double rechit_Energy_RPD(const QIE10DataFrame& digi);
  double rechit_Time(const QIE10DataFrame& digi);
  double rechit_TDCtime(const QIE10DataFrame& digi);
  double rechit_ChargeWeightedTime(const QIE10DataFrame& digi);
  double rechit_EnergySOIp1(const QIE10DataFrame& digi);
  double rechit_RatioSOIp1(const QIE10DataFrame& digi);
  int rechit_Saturation(const QIE10DataFrame& digi);
  


private:

std::vector<double> mValues;
unsigned int nbins_ = 64;

double exact_offsets[16] = {-0.50000, -0.66002, 18.71833, -270.46150, -0.50000, -0.66002, 18.71833, -270.46150,
-0.50000, -0.66002, 18.71833, -270.46150, -0.50000, -0.66002, 18.71833, -270.46150}; 
double exact_slopes[16] = { 0.30683, 0.31046, 0.31191, 0.31766, 0.30683, 0.31046, 0.31191, 0.31766,
0.30683, 0.31046, 0.31191, 0.31766, 0.30683, 0.31046, 0.31191, 0.31766};

const double binMin2[64] = {-0.5,  0.5,   1.5,   2.5,   3.5,   4.5,   5.5,   6.5,   7.5,   8.5,   9.5,
                     10.5,  11.5,  12.5,  13.5,  14.5,  // 16 bins with width 1x
                     15.5,  17.5,  19.5,  21.5,  23.5,  25.5,  27.5,  29.5,  31.5,  33.5,  35.5,
                     37.5,  39.5,  41.5,  43.5,  45.5,  47.5,  49.5,  51.5,  53.5,  // 20 bins with width 2x
                     55.5,  59.5,  63.5,  67.5,  71.5,  75.5,  79.5,  83.5,  87.5,  91.5,  95.5,
                     99.5,  103.5, 107.5, 111.5, 115.5, 119.5, 123.5, 127.5, 131.5, 135.5,  // 21 bins with width 4x
                     139.5, 147.5, 155.5, 163.5, 171.5, 179.5, 187.5};                      // 7 bins with width 8x
                     
                     
double center(unsigned fAdc){
  if (fAdc < 4 * nbins_) {
    if (fAdc % nbins_ == nbins_ - 1)
      return 0.5 * (3 * mValues[fAdc] - mValues[fAdc - 1]);  // extrapolate
    else
      return 0.5 * (mValues[fAdc] + mValues[fAdc + 1]);  // interpolate
  }
  return 0.;
}

  unsigned range_(unsigned fAdc) {
    //6 bit mantissa in QIE10, 5 in QIE8
    return (nbins_ == 32) ? (fAdc >> 5) & 0x3 : (fAdc >> 6) & 0x3;
  }

void expand() {
  int scale = 1;
  for (unsigned range = 1; range < 4; range++) {
    int factor = nbins_ == 32 ? 5 : 8;  // QIE8/QIE10 -> 5/8
    scale *= factor;
    unsigned index = range * nbins_;
    unsigned overlap = (nbins_ == 32) ? 2 : 3;  // QIE10 -> 3 bin overlap
    mValues[index] = mValues[index - overlap];  // link to previous range
    for (unsigned i = 1; i < nbins_; i++) {
      mValues[index + i] = mValues[index + i - 1] + scale * (mValues[i] - mValues[i - 1]);
    }
  }
  mValues[nbins_ * 4] = 2 * mValues[nbins_ * 4 - 1] - mValues[nbins_ * 4 - 2];  // extrapolate
}

bool setLowEdge(double fValue, unsigned fAdc) {
  if (fAdc >= nbins_)
    return false;
  mValues[fAdc] = fValue;
  return true;
}

bool setLowEdges(unsigned int nbins_, const double *fValue) {
  mValues.clear();
  mValues.resize(4 * nbins_ + 1);
  bool result = true;
  for (unsigned int adc = 0; adc < nbins_; adc++)
    result = result && setLowEdge(fValue[adc], adc);
  expand();
  return result;
}

double RPD_Peds[2][16] = {{91.09000, 78.96000, 72.24000, 80.10000,
102.30000, 91.76000, 97.17000, 134.89999,
144.30000, 123.90000, 129.20000, 140.60001,
78.42000, 84.01000, 69.50000, 67.18000},
{55.32000, 45.91000, 36.20000, 38.67000,
38.23000, 36.85000, 32.18000, 35.90000,
34.93000, 36.02000, 30.25000, 41.78000,
70.23000, 59.18000, 59.21000, 63.49000}
};
double RPD_PedWidths[2][16] = {{124.70000, 113.20000, 97.31000, 110.10000,
144.89999, 127.20000, 139.00000, 190.70000,
207.39999, 179.1000, 185.20000, 202.39999,
105.60000, 113.90000, 92.76000, 87.8200},
{116.3000, 119.00000, 45.24000, 76.88000,
88.19000, 74.40000, 66.26000, 59.59000,
70.50000, 72.09000, 54.09000, 74.78000,
129.00000, 102.4000, 89.08000, 112.0000}
};

double RPD_Gains[2][16] = {{1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0},
{1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0,
1.0, 1.0, 1.0, 1.0}
};


double EM_Peds[2][5] = {{120.900, 118.200, 265.400, 113.600, 126.100},
{136.700, 91.010, 148.000, 78.600, 144.600}
};
double EM_PedWidths[2][5] = {{173.600, 161.400, 425.900, 165.200, 196.000},
{170.700, 112.900, 198.300, 95.610, 193.800}
};

double EM_Gains[2][5] = {{0.0573900, 0.0573900, 0.0573900, 0.0573900, 0.0573900},
{0.1026100, 0.1026100, 0.1026100, 0.1026100, 0.1026100}
};

double HAD_Peds[2][4] = {{181.400, 143.100, 136.800, 127.700},
{54.670, 47.520, 60.920, 64.960}
};
double HAD_PedWidths[2][4] = {{252.000, 199.500, 187.600, 183.600},
{62.620, 55.460, 72.370, 81.590}
};

double HAD_Gains[2][4] = {{0.5739000, 0.5739000, 0.5739000, 0.5739000},
{1.0261000, 1.0261000, 1.0261000, 1.0261000}
};

int nTs_ = 6;
int signalTs_ = 2;
int noiseTs_ = 1; 
float ootpuFrac = 97.0/256.0;
int maxValue_ = 255;

inline double subPedestal(const float charge, const float ped, const float width) {
 if (charge - ped > width)
   return (charge - ped);
 else
   return (0);
}

};

#endif
