#ifndef gen_ExternalDecays_TauolaWrapper_h
#define gen_ExternalDecays_TauolaWrapper_h

/********** TauolaWrapper
 *
 * Wrapper to Fortran functions in TAUOLA tau decay library
 *
 * Christian Veelken
 * 04/17/07
 *
 * Modified to contain access to individual decays (COMMON TAUBRA), and
 * force polarization like in case of particle gun taus (SUBROUTINE DEXAY
 * and COMMON MOMDEC)
 * 23.2.2009/S.Lehti
 *
 * Modified to remove everything related to pythia6 or pretauola;
 * all pythia6-related functionalities and/or ties are moving back
 * to Pythia6Interface;
 * the concept will ensure transparent use of Tauola with ANY
 * multi-purpose generator
 * J.V.Yarba, Feb.26, 2009
 *
 ***************************************/

//
//-------------------------------------------------------------------------------
//

// main function(s) of TAUOLA/pretauola tau decay library

extern "C" {
  void tauola_(int*, int*);
  void tauola_srs_(int*,int*);
  void taurep_(int*);
  void ranmar_(float*,int*);
  void rmarin_(int*, int*, int*);
}
#define tauola tauola_

void inline call_tauola (int mode, int polarization) { tauola(&mode, &polarization); }

extern "C" {
  extern void dexay_(int*, float[4]);
}
#define dexay dexay_

void inline call_dexay (int mode, float polarization[4]) { dexay(&mode, polarization); }

//
//-------------------------------------------------------------------------------
//

// common block with steering parameters for CMS specific Fortran interface to TAUOLA

extern "C" {
  extern struct {
    int pjak1;
    int pjak2;
    int mdtau;
  } ki_taumod_;
}
#define ki_taumod ki_taumod_

extern "C" {
  extern struct {
    int jak1;
    int jak2;
    int itdkrc;
    int ifphot;
    int ifhadm;
    int ifhadp;
  } libra_ ;
}
#define libra libra_

extern "C" {
  extern struct {
    float gamprt[30];
    int jlist[30];
    int nchan;
  } taubra_;
}
#define taubra taubra_

extern "C" {
  extern struct {
    double q1[4];
    double q2[4];
    double p1[4];
    double p2[4];
    double p3[4];
    double p4[4];
  } momdec_;
}
#define momdec momdec_

extern "C" {
  extern struct {
    int np1;
    int np2;
  } taupos_;
}
#define taupos taupos_

#endif
