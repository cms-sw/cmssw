#define MAXMOD 56 // 50 real channels + 6 dump channels
#define NMOD 50 // 50 real channels: (9 ZDC + 16 RPD) x 2 sides
#define MAXTS 10 // 10 time slices before 2024
#define NTS 6 // 6 time slices since 2024

struct MyZDCRecHit {
  int n;
  int zside[MAXMOD];
  int section[MAXMOD];
  int channel[MAXMOD];
  float energy[MAXMOD];
  float time[MAXMOD];
  float TDCtime[MAXMOD];
  float chargeWeightedTime[MAXMOD];
  float energySOIp1[MAXMOD];
  float ratioSOIp1[MAXMOD];
  int saturation[MAXMOD];

  float sumPlus;
  float sumMinus;
  float sumPlus_Aux;
  float sumMinus_Aux;
};

struct MyZDCDigi {
  int n;
  float chargefC[MAXTS][MAXMOD];
  int adc[MAXTS][MAXMOD];
  int tdc[MAXTS][MAXMOD];
  int zside[MAXMOD];
  int section[MAXMOD];
  int channel[MAXMOD];

  // for debug usage, and the sum should be from rechits
  float sumPlus;
  float sumMinus;
};

