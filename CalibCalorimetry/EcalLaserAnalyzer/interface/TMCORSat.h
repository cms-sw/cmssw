#ifndef TMCORSat_H
#define TMCORSat_H

#include "TObject.h"

class TMCORSat: public TObject 
{

 private:
	
  int smin;
  float pn_rg[36][10];
  float apd_rg[36][2][1700];

  int convert(int);

  void init();
  void loadConsts();
  void loadConsts(int);

 public:
  // Default Constructor, mainly for Root
  TMCORSat();
  TMCORSat(int);

  // Destructor: Does nothing
  virtual ~TMCORSat();

  double getPNrg(int, int, int);
  double getAPDrg(int, int ,int);

  //  ClassDef(TMCORSat,1)
};

#endif

