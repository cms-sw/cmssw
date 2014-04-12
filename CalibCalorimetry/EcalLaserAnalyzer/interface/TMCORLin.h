#ifndef TMCORLin_H
#define TMCORLin_H

#include "TObject.h"

class TMCORLin: public TObject 
{

 private:
	
  int smin;
  float pn_par0[36][2][10];
  float pn_par1[36][2][10];
  float pn_par2[36][2][10];
  float apd_par0[36][3][1700];
  float apd_par1[36][3][1700];
  float apd_par2[36][3][1700];

  int convert(int);

  void init();
  void loadConsts();
  void loadConsts(int,int);

 public:
  // Default Constructor, mainly for Root
  TMCORLin();
  TMCORLin(int);

  // Destructor: Does nothing
  virtual ~TMCORLin();

  double computeCorlin_pn(int, int ,int ,double );
  double computeCorlin_apd(int, int ,int ,double );

  //  ClassDef(TMCORLin,1)
};

#endif

