#ifndef CALIBTRACKER_SISTRIPCHANNELGAIN_STAPVGAIN_H
#define CALIBTRACKER_SISTRIPCHANNELGAIN_STAPVGAIN_H

class TH1F;

struct stAPVGain{
  unsigned int Index; 
  int          Bin;
  unsigned int DetId;
  unsigned int APVId;
  unsigned int SubDet;
  float        x;
  float        y;
  float        z;
  float        Eta;
  float        R;
  float        Phi;
  float        Thickness;
  double       FitMPV;
  double       FitMPVErr;
  double       FitWidth;
  double       FitWidthErr;
  double       FitChi2;
  double       FitNorm;
  double       Gain;
  double       CalibGain;
  double       PreviousGain;
  double       PreviousGainTick;
  double       NEntries;
  TH1F*        HCharge;
  TH1F*        HChargeN;
  bool         isMasked;
};

enum statistic_type {None=-1, StdBunch, StdBunch0T, FaABunch, FaABunch0T, IsoBunch, IsoBunch0T, Harvest};

#endif
