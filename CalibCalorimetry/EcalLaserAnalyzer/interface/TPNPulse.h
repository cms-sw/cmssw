#ifndef TPNPulse_H
#define TPNPulse_H

#include "TObject.h"
#include<vector>

class TPNPulse: public TObject 
{

 private:

  int _nsamples;
  int _presample;
  

  double *adc_;
  bool isMaxFound_;
  bool isPedCalc_;
  double adcMax_;
  int iadcMax_;
  double pedestal_;

  void init(int, int );

 public:


  // Default Constructor, mainly for Root
  TPNPulse();

  // Constructor
  TPNPulse(int, int);

  // Destructor: Does nothing
  virtual ~TPNPulse();

  bool setPulse(double*);
  double getMax();
  int getMaxSample();
  double getPedestal();
  double* getAdcWithoutPedestal();
  void setPresamples(int);
  //ClassDef(TPNPulse,1)
};

#endif
