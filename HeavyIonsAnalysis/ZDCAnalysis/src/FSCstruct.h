#define MAXMOD 56 // 50 real channels + 6 dump channels
#define NMOD 50 // 50 real channels: (9 ZDC + 16 RPD + 6 FSC) x 2 sides
#define MAXTS 10 // 10 time slices before 2024
#define NTS 6 // 6 time slices since 2024
#define NFSC 6 // 6 FSC detectors on each side 


struct MyFSCDigi {
  int n;
  float chargefC[MAXTS][MAXMOD];
  int adc[MAXTS][MAXMOD];
  int tdc[MAXTS][MAXMOD];
  int zside[MAXMOD];
  int section[MAXMOD];
  int channel[MAXMOD];

  float charge[MAXMOD];
  float charge_bare[MAXMOD];
  float Fitted_QTS0[MAXMOD];
  float Fitted_QTS2[MAXMOD];
  int saturation[MAXMOD]; // 0 is normal, 1 ADCTS2 saturated, 2 ADCTS3 saturated, +10 ADCTS0 saturated

  float sumPlus;
  float sumMinus;
  float sumPlus_FSC2only;
  float sumPlus_FSC3only;
  float sumMinus_FSC2only;
  float sumMinus_FSC3only;
};

