/*
 *
 * Wrapper for FORTRAN version of HYDJET
 *
 * Camelia Mironov
 *
 */


// HYDJET routine declaration

#define _MAXMULsize_ 150000

#define HYDRO hydro_
extern "C" {
  void HYDRO(double& a,int& ifb,double& bmin,double& bmax,double& bfix,int& nh);
}


extern "C" {
  extern struct{
    int nnhyd;
    int khyd[5][_MAXMULsize_];
    float phyd[5][_MAXMULsize_];
    float vhyd[5][_MAXMULsize_];
  }hyd_;
}
#define hyd hyd_

extern "C" {
  extern struct {
    float ytfl;
    float ylfl;
    float fpart;
  }hyflow_;
}
#define hyflow hyflow_


extern "C" {
  extern struct{
    float bgen;
    int nbcol;
    int npart;
    int npyt;
    int nhyd;
  }hyfpar_;
}
#define hyfpar hyfpar_



extern "C" {
  extern struct{
    int nl;
    int kl[5][_MAXMULsize_];
    float pl[5][_MAXMULsize_];
    float vl[5][_MAXMULsize_];
  }hyjets_;
}
#define hyjets hyjets_


extern "C" {
  extern struct{
    int nhsel;
    float ptmin;
    float njet;
  }hyjpar_;
}
#define hyjpar hyjpar_


extern "C" {
  extern struct{
    int n;
    int k[5][_MAXMULsize_];
    float p[5][_MAXMULsize_];
    float v[5][_MAXMULsize_];
  }lujets_;
}
#define lujets lujets_

