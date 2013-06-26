#ifndef GeneratorInterface_HydjetInterface_HydjetWrapper
#define GeneratorInterface_HydjetInterface_HydjetWrapper

//
// $Id: HydjetWrapper.h,v 1.6 2007/12/04 03:50:39 mironov Exp $
//

/*
 *
 * Wrapper for FORTRAN version of HYDJET
 *
 * Camelia Mironov
 *
 */

extern "C" {
   void hyinit_(double& energy,double& a,int& ifb1,double& bmin,double& bmax,double& bfix1,int& nh1);
}
#define HYINIT hyinit_

#define _MAXMULsize_ 150000


extern "C" {
  void hyevnt_();
}
#define HYEVNT hyevnt_


extern "C" {
  extern struct {
    double ytfl;
    double ylfl;
    double Tf;
    double fpart;
  } hyflow_;
}
#define hyflow hyflow_


extern "C" {
  extern struct{
    double bgen;
    double nbcol;
    double npart;
    int npyt;
    int nhyd;
  }hyfpar_;
}
#define hyfpar hyfpar_

extern "C" {
  extern struct {
    double bminh;
    double bmaxh;
    double AW;
    double RA;
    double npar0;
    double nbco0;
    double Apb;
    double Rpb;
    int   np;
    int   init;
    int   ipr;
  
  } hyipar_;
}
#define hyipar	hyipar_


extern "C" {
  extern struct{
    int nhj;
    int nhp;
    int khj[5][_MAXMULsize_];
    double phj[5][_MAXMULsize_];
    double vhj[5][_MAXMULsize_];
  }hyjets_;
}
#define hyjets hyjets_


extern "C" {
  extern struct{

    double ptmin;
    double sigin;
    double sigjet;
    int nhsel;
    int ishad;
    int njet;
 
  }hyjpar_;
}
#define hyjpar hyjpar_


extern "C" {
   extern struct {
     int mrlu[6];
     int rrlu[100];
   } ludatr_;
}
#define ludatr ludatr_


extern "C" {
  extern struct{
    double T0u;
    double tau0u;
    int   nfu;
    int   ienglu;
    int   ianglu;
  }pyqpar_;
}
#define pyqpar pyqpar_

#endif
