#ifndef GeneratorInterface_HydjetInterface_HydjetWrapper
#define GeneratorInterface_HydjetInterface_HydjetWrapper

//
// $Id: HydjetWrapper.h,v 1.2 2007/04/28 16:08:16 mballint Exp $
//

/*
 *
 * Wrapper for FORTRAN version of HYDJET
 *
 * Camelia Mironov
 *
 */


#define _MAXMULsize_ 150000

extern "C" {
  void hydro_(float& a,int& ifb,float& bmin,float& bmax,float& bfix,int& nh);
}
#define HYDRO hydro_


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
  } hyflow_;
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


// common /hyipar/ bminh,bmaxh,AW,RA,sigin,np  

extern "C" {
  extern struct {
    float	bminh;
	 float	bmaxh;
	 float	AW;
	 float	RA;
	 float	sigin;
	 float	np;
  } hyipar_;
}
#define hyipar	hyipar_


extern "C" {
   extern struct {
     int mrlu[6];
     int rrlu[100];
   } ludatr_;
}
#define ludatr ludatr_

#endif
