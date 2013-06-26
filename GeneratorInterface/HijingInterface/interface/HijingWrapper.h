#ifndef GeneratorInterface_HijingInterface_HijingWrapper
#define GeneratorInterface_HijingInterface_HijingWrapper

/*
 * Wrapper for FORTRAN version of HIJING 
 */

#define _MAXNUMPARTICLE_ 130000

extern "C" {
  void hijset_(float& efrm, const char* frame, const char* proj, const char* targ, int& iap, int& izp, int& iat, int& izt, int, int, int); 
}
#define HIJSET hijset_

extern "C" {
  void hijing_(const char* frame, float& bmin0, float& bmax0, int);
}
#define HIJING hijing_

extern "C" {
  extern struct{ 
    int natt;
    int eatt;
    int jatt;
    int nt;
    int np;
    int n0;
    int n01;
    int n10;
    int n11;
  }himain1_;
}
#define himain1 himain1_

extern "C" {
  extern struct{ 
     int katt[4][_MAXNUMPARTICLE_];
     float patt[4][_MAXNUMPARTICLE_];
     float vatt[4][_MAXNUMPARTICLE_];
  }himain2_;
}
#define himain2 himain2_

extern "C" {
  extern struct{ 
    float  hipr1[100];
    int    ihpr2[50];
    float  hint1[100];
    int    ihnt2[50];
  }hiparnt_;
}
#define hiparnt hiparnt_



#endif
