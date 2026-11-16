#ifndef HDECAY_HH
#define HDECAY_HH

extern "C" {

extern struct {
  double amsm;
  double ama;
  double aml;
  double amh;
  double amch;
} hmass_;
/*
  extern struct{
    double amchi;
  }chimass_;
  */
extern struct {
  double ams;
  double amc;
  double amb;
  double amt;
} masses_;

extern struct {
  double xlambda;
  double amc0;
  double amb0;
  double amt0;
  int n0;
} als_;

extern struct {
  double gf;
  double alph;
  double amtau;
  double ammuon;
  double amz;
  double amw;
} param_;

extern struct {
  double vus;
  double vcb;
  double vub;
} ckmpar_;

extern struct {
  double gamc0;
  double gamt0;
  double gamt1;
  double gamw;
  double gamz;
} wzwdth_;

extern struct {
  int ionsh;
  int ionwz;
  int iofsusy;
} onshell_;

extern struct {
  int nfgg;
} oldfash_;

extern struct {
  int ihiggs;
  int nnlo;
  int ipole;
} flag_;

extern struct {
  double smbrb;
  double smbrl;
  double smbrm;
  double smbrs;
  double smbrc;
  double smbrt;
  double smbrg;
  double smbrga;
  double smbrzga;
  double smbrw;
  double smbrz;
  double smwdth;
} widthsm_;

/*
  extern struct{
    double amneut;
    double xmneut;
    double amchar;
    double amst;
    double amsb;
    double amsl;
    double amsu;
    double amsd;
    double amse;
    double amsn;
  }smass_;
  */
extern struct {
  double gat;
  double gab;
  double glt;
  double glb;
  double ght;
  double ghb;
  double gzah;
  double gzal;
  double ghhh;
  double glll;
  double ghll;
  double glhh;
  double ghaa;
  double glaa;
  double glw;
  double ghvv;
  double glpm;
  double ghpm;
  double b;
  double a;
} coup_;

extern struct {
  double amsb;
} strange_;

double xitla_(int*, double*, double*);
void bernini_(int*);
//fix unused parameter warning
//void hdec_(double*);
void hdec_();
void alsini_(double*);
}

#endif
