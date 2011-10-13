#ifndef TPNFit_H
#define TPNFit_H

#include "TObject.h"

#define NMAXSAMP2 50

class TPNFit: public TObject 
{

 private:	

  int fNsamples ;
  int fNum_samp_bef_max;
  int fNum_samp_after_max;

  int firstsample,lastsample;
  double t[NMAXSAMP2],val[NMAXSAMP2];
  double fv1[NMAXSAMP2],fv2[NMAXSAMP2],fv3[NMAXSAMP2];
  double ampl;
  double timeatmax;
  

 public:
  // Default Constructor, mainly for Root
  TPNFit();

  // Destructor: Does nothing
  virtual ~TPNFit();

  // Initialize 
  void init(int,int,int);

  double doFit(int,double *);
  double getAmpl() {return ampl;}
  double getTimax() {return timeatmax;}

  //  ClassDef(TPNFit,1)
};

#endif



