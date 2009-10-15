#ifndef GeneratorInterface_AMPTInterface_AMPTWrapper
#define GeneratorInterface_AMPTInterface_AMPTWrapper

/*
 * Wrapper for FORTRAN version of AMPT 
 */

//gsfs changed to 150001
#define _MAXNUMPARTICLE_ 150001

extern "C" {
  void amptset_(double& efrm, const char* frame, const char* proj, const char* targ, int& iap, int& izp, int& iat, int& izt, int, int, int); 
}
#define AMPTSET amptset_

extern "C" {
  void ampt_(const char* frame, double& bmin0, double& bmax0, int);
}
#define AMPT ampt_
//gsfs changed entries to agree with calling sequence in AMPT
extern "C" {
  extern struct{ 
    float eatt;
    int jatt;
    int natt;
    int nt;
    int np;
    int n0;
    int n01;
    int n10;
    int n11;
  }hmain1_;
}
#define hmain1 hmain1_

extern "C" {
  extern struct{ 
    int katt[4][_MAXNUMPARTICLE_];
//gsfs changed following to float from double
    float patt[4][_MAXNUMPARTICLE_];
  }hmain2_;
}
#define hmain2 hmain2_

extern "C" {
  extern struct{ 
    float  hipr1[100];
    int    ihpr2[50];
    float  hint1[100];
    int    ihnt2[50];
  }hparnt_;
}
#define hparnt hparnt_

#endif
