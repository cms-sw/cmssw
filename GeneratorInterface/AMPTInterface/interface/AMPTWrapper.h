#ifndef GeneratorInterface_AMPTInterface_AMPTWrapper
#define GeneratorInterface_AMPTInterface_AMPTWrapper

/*
 * Wrapper for FORTRAN version of AMPT 
 */

//gsfs changed to 150001
#define _MAXNUMPARTICLE_ 150001

extern "C" {
void amptset_(double& efrm,
              const char* frame,
              const char* proj,
              const char* targ,
              int& iap,
              int& izp,
              int& iat,
              int& izt,
              int,
              int,
              int);
}
#define AMPTSET amptset_

extern "C" {
void ampt_(const char* frame, double& bmin0, double& bmax0, int);
}
#define AMPT ampt_

extern "C" {
int invflv_(int&);
}
#define INVFLV invflv_

//gsfs changed entries to agree with calling sequence in AMPT
extern "C" {
extern struct {
  float eatt;
  int jatt;
  int natt;
  int nt;
  int np;
  int n0;
  int n01;
  int n10;
  int n11;
} hmain1_;
}
#define hmain1 hmain1_

extern "C" {
extern struct {
  int lblast[_MAXNUMPARTICLE_];
  float xlast[_MAXNUMPARTICLE_][4];
  float plast[_MAXNUMPARTICLE_][4];
  int nlast;
} hbt_;
}
#define hbt hbt_

extern "C" {
extern struct {
  float hipr1[100];
  int ihpr2[50];
  float hint1[100];
  int ihnt2[50];
} hparnt_;
}
#define hparnt hparnt_

extern "C" {
extern struct {
  int mstu[200];
  float paru[200];
  int mstj[200];
  float parj[200];
} ludat1_;
}
#define ludat1 ludat1_

extern "C" {
extern struct {
  int nevent;
  int isoft;
  int isflag;
  int izpc;
} anim_;
}
#define anim anim_

extern "C" {
extern struct {
  float dpcoal;
  float drcoal;
  float ecritl;
} coal_;
}
#define coal coal_

extern "C" {
extern struct {
  float xmp;
  float xmu;
  float alpha;
  float rscut2;
  float cutof2;
} para2_;
}
#define para2 para2_

extern "C" {
extern struct {
  int ioscar;
  int nsmbbbar;
  int nsmmeson;
} para7_;
}
#define para7 para7_

extern "C" {
extern struct {
  int idpert;
  int npertd;
  int idxsec;
} para8_;
}
#define para8 para8_

extern "C" {
extern struct {
  float masspr;
  float massta;
  int iseed;
  int iavoid;
  float dt;
} input1_;
}
#define input1 input1_

extern "C" {
extern struct {
  int ilab;
  int manyb;
  int ntmax;
  int icoll;
  int insys;
  int ipot;
  int mode;
  int imomen;
  int nfreq;
  int icflow;
  int icrho;
  int icou;
  int kpoten;
  int kmul;
} input2_;
}
#define input2 input2_

extern "C" {
extern struct {
  int nsav;
  int iksdcy;
} resdcy_;
}
#define resdcy resdcy_

extern "C" {
extern struct {
  int iphidcy;
  float pttrig;
  int ntrig;
  int maxmiss;
} phidcy_;
}
#define phidcy phidcy_

extern "C" {
extern struct {
  int iembed;
  float pxqembd;
  float pyqembd;
  float xembd;
  float yembd;
} embed_;
}
#define embed embed_

extern "C" {
extern struct {
  int ipop;
} popcorn_;
}
#define popcorn popcorn_

#endif
