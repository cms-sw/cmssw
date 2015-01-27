#ifndef HYJETCOMMON
#define HYJETCOMMON

extern "C" {

#define f2cFortran
#include "cfortran.h"


  //----------------------------------------------------------------
  // common /hyipar/ bminh,bmaxh,AW,RA,npar0,nbco0,Apb,Rpb,np,init,ipr        
  typedef struct //HYIPAR
  {
    double bminh;
    double bmaxh; 
    double AW;
    double RA;
    double npar0;
    double nbco0;
    double Apb;
    double Rpb;
    double np;
    int init;
    int ipr;
  }HYIPARCommon;
 
#define HYIPAR COMMON_BLOCK(HYIPAR,hyipar)
  COMMON_BLOCK_DEF(HYIPARCommon, HYIPAR);
  //----------------------------------------------------------------

  //      common/service/iseed_fromC,iPythDecay,parPYTH(100)
  typedef struct //SERVICE
  {
    int iseed_fromC; 
    int iPythDecay;
    double charm;
  }SERVICECommon;
 
#define SERVICE COMMON_BLOCK(SERVICE,service)
  COMMON_BLOCK_DEF(SERVICECommon, SERVICE);
  //----------------------------------------------------------------

  //  common/SERVICEEV/ipdg,delta

  typedef struct //SERVICEEV
  {
    float psiv3;
    float delta;
    int KC;
    int ipdg;
  }SERVICEEVCommon;
 
#define SERVICEEV COMMON_BLOCK(SERVICEEV,serviceev)
  COMMON_BLOCK_DEF(SERVICEEVCommon, SERVICEEV);

  //----------------------------------------------------------------

  // common /hyjpar/ ptmin,sigin,sigjet,nhsel,iPyhist,ishad,njet 
  typedef struct //HYJPAR
  {
    double ptmin;
    double sigin;
    double sigjet;
    int nhsel;
    int iPyhist;
    int ishad;
    int njet;
  }HYJPARCommon;
 
#define HYJPAR COMMON_BLOCK(HYJPAR,hyjpar)
  COMMON_BLOCK_DEF(HYJPARCommon, HYJPAR);
  //----------------------------------------------------------------


  //      common /hypyin/ ene,rzta,rnta,bfix,ifb,nh
  typedef struct //HYPYIN
  {
    double ene;
    double rzta;
    double rnta;
    double bfix; 
    int ifb;
    int nh;
  }HYPYINCommon;
 
#define HYPYIN COMMON_BLOCK(HYPYIN,hypyin)
  COMMON_BLOCK_DEF(HYPYINCommon, HYPYIN);


  //----------------------------------------------------------------
  //  common /hyfpar/ bgen,nbcol,npart,npyt,nhyd,npart0        
  typedef struct //HYFPAR
  {
    double bgen;
    double nbcol;
    double npart;
    double npart0;
    int npyt;
    int nhyd;
  }HYFPARCommon;
 
#define HYFPAR COMMON_BLOCK(HYFPAR,hyfpar)
  COMMON_BLOCK_DEF(HYFPARCommon, HYFPAR);

  //----------------------------------------------------------------
  typedef struct //HYPART
  {
    double ppart[150000][20];
    double bmin;
    double bmax;
    int njp;
  }HYPARTCommon;
 
#define HYPART COMMON_BLOCK(HYPART,hypart)
  COMMON_BLOCK_DEF(HYPARTCommon, HYPART);
  //----------------------------------------------------------------

  //      common /pyqpar/ T0,tau0,nf,ienglu,ianglu 

  typedef struct //PYQPAR
  {
    double T0;
    double tau0;
    int nf;
    int ienglu; 
    int ianglu;
  }PYQPARCommon;
 
#define PYQPAR COMMON_BLOCK(PYQPAR,pyqpar)
  COMMON_BLOCK_DEF(PYQPARCommon, PYQPAR);

} 
#endif                     
