#ifndef TAPDPulse_H
#define TAPDPulse_H

#include "TObject.h"
#include<vector>

class TAPDPulse: public TObject 
{

 private:

  int _nsamples;
  int _presample;
  int _firstsample;
  int _lastsample;
  int _timingcutlow;
  int _timingcuthigh;
  int _timingquallow;
  int _timingqualhigh;
  double _ratiomaxcutlow;
  double _ratiomincutlow;
  double _ratiomincuthigh;
  

  double *adc_;
  bool isMaxFound_;
  bool isPedCalc_;
  double adcMax_;
  int iadcMax_;
  double pedestal_;

  void init(int, int, int, int, int, int, int, int, double, double, double );

 public:


  // Default Constructor, mainly for Root
  TAPDPulse();

  // Constructor
  TAPDPulse(int, int, int, int, int, int, int, int, double, double, double);

  // Destructor: Does nothing
  virtual ~TAPDPulse();

  bool setPulse(double*);
  double getMax();
  int getMaxSample();
  double getDelta(int, int);
  double getRatio(int, int);
  bool isTimingOK();
  bool isTimingQualOK();
  bool areFitSamplesOK();
  bool isPulseOK();
  bool arePulseRatioOK();
  bool isPulseRatioMaxOK();
  bool isPulseRatioMinOK();
  double getPedestal();
  double* getAdcWithoutPedestal();
  void setPresamples(int);
  //ClassDef(TAPDPulse,1)
};

#endif
