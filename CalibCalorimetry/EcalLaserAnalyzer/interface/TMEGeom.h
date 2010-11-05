#ifndef TMEGeom_H
#define TMEGeom_H

#include "TObject.h"

#define nTT 25

class TMEGeom: public TObject 
{

 private:

 int ttindarr[5][5];


 public:
  // Default Constructor, mainly for Root
  TMEGeom();

  // Destructor: Does nothing
  virtual ~TMEGeom();

int nbOfXTalinmodN(int);
int nbOfXTalinlmodN(int);

int xtaltoadcn(int);
int adcltoxtal(int,int);
int adcltoadcn(int,int);
int adcltotNumb(int,int);
int adcmtoadcn(int,int);
int adcmtoadcl(int,int,int);
int adcntoadcm(int);
int adcntomodN(int);
int adcntolmodN(int,int);
int adcntoxtal(int);
int tNumbtomodN(int);
int tNumbtomodulN(int);
int tNumbtolmodN(int);
int tNumbtoside(int);
int lmodNtoside(int);
int lmodNtomodN(int);
int lmodNtolmcha(int);

int modN_offset(int);
int lmodN_offset(int);

int adcntoij(int);
int adcltoij(int,int);
int ijtoadcn(int,int);
int ijtoadcl(int,int,int);

int tNumbtolvcha(int);
int tNumbtohvcha(int);
int adcltolvcha(int,int);
int adcltohvcha(int,int);

int hvchatolvcha(int);

void tNumbtoij(int);

//  ClassDef(TMEGeom,1)
};

#endif
