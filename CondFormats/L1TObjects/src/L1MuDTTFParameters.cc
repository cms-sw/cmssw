//-------------------------------------------------
//
//   Class: L1MuDTTFParameters
//
//   Description: DTTF Parameters from OMDS
//
//
//   $Date: 2008/02/25 15:26:57 $
//   $Revision: 1.2 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"

//---------------
// C++ Headers --
//---------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------


// --------------------------------
//       class L1MuDTTFParameters
//---------------------------------

//--------------
// Operations --
//--------------

//
// reset extrapolation look-up tables
//
void L1MuDTTFParameters::reset() {

  for( int i=0; i<6; i++ ) {
    for( int j=0; j<12; j++ ) {

      inrec_chdis_st1[i][j]  = false;
      inrec_chdis_st2[i][j]  = false;
      inrec_chdis_st3[i][j]  = false;
      inrec_chdis_st4[i][j]  = false;
      inrec_qual_st1[i][j]   = 0;
      inrec_qual_st2[i][j]   = 0;
      inrec_qual_st3[i][j]   = 0;
      inrec_qual_st4[i][j]   = 0;
      soc_stdis_n[i][j]      = 0;
      soc_stdis_wl[i][j]     = 0;
      soc_stdis_wr[i][j]     = 0;
      soc_stdis_zl[i][j]     = 0;
      soc_stdis_zr[i][j]     = 0;
      soc_qcut_st1[i][j]     = 0;
      soc_qcut_st2[i][j]     = 0;
      soc_qcut_st4[i][j]     = 0;
      soc_qual_csc[i][j]     = 0;
      soc_run_21[i][j]       = false;
      soc_nbx_del[i][j]      = false;
      soc_csc_etacanc[i][j]  = false;
      soc_openlut_extr[i][j] = false;
    }
  }
}

void L1MuDTTFParameters::set_inrec_chdis_st1(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_st1[check(wh,sc)][sc] = val;
}

bool L1MuDTTFParameters::get_inrec_chdis_st1(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_st1[check(wh,sc)][sc];
}

void L1MuDTTFParameters::set_inrec_chdis_st2(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_st2[check(wh,sc)][sc] = val;
}

bool L1MuDTTFParameters::get_inrec_chdis_st2(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_st2[check(wh,sc)][sc];
}

void L1MuDTTFParameters::set_inrec_chdis_st3(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_st3[check(wh,sc)][sc] = val;
}

bool L1MuDTTFParameters::get_inrec_chdis_st3(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_st3[check(wh,sc)][sc];
}

void L1MuDTTFParameters::set_inrec_chdis_st4(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_st4[check(wh,sc)][sc] = val;
}

bool L1MuDTTFParameters::get_inrec_chdis_st4(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_st4[check(wh,sc)][sc];
}

void L1MuDTTFParameters::set_inrec_qual_st1(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_qual_st1[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_inrec_qual_st1(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (inrec_qual_st1[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_inrec_qual_st2(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_qual_st2[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_inrec_qual_st2(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (inrec_qual_st2[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_inrec_qual_st3(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_qual_st3[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_inrec_qual_st3(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (inrec_qual_st3[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_inrec_qual_st4(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_qual_st4[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_inrec_qual_st4(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (inrec_qual_st4[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_stdis_n(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_stdis_n[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_stdis_n(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_stdis_n[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_stdis_wl(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_stdis_wl[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_stdis_wl(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_stdis_wl[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_stdis_wr(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_stdis_wr[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_stdis_wr(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_stdis_wr[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_stdis_zl(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_stdis_zl[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_stdis_zl(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_stdis_zl[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_stdis_zr(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_stdis_zr[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_stdis_zr(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_stdis_zr[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_qcut_st1(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_qcut_st1[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_qcut_st1(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_qcut_st1[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_qcut_st2(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_qcut_st2[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_qcut_st2(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_qcut_st2[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_qcut_st4(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_qcut_st4[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_qcut_st4(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_qcut_st4[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_qual_csc(int wh, int sc, const unsigned short int val) {

  if ( check(wh,sc) == -99 ) return;
  soc_qual_csc[check(wh,sc)][sc] = val&0x7;
}

unsigned short int L1MuDTTFParameters::get_soc_qual_csc(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return 0;
  return (soc_qual_csc[check(wh,sc)][sc])&0x7;
}

void L1MuDTTFParameters::set_soc_run_21(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  soc_run_21[check(wh,sc)][sc] = val;
}

bool L1MuDTTFParameters::get_soc_run_21(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return soc_run_21[check(wh,sc)][sc];
}

void L1MuDTTFParameters::set_soc_nbx_del(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  soc_nbx_del[check(wh,sc)][sc] = val;
}

bool L1MuDTTFParameters::get_soc_nbx_del(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return soc_nbx_del[check(wh,sc)][sc];
}

void L1MuDTTFParameters::set_soc_csc_etacanc(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  soc_csc_etacanc[check(wh,sc)][sc] = val;
}

bool L1MuDTTFParameters::get_soc_csc_etacanc(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return soc_csc_etacanc[check(wh,sc)][sc];
}

void L1MuDTTFParameters::set_soc_openlut_extr(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  soc_openlut_extr[check(wh,sc)][sc] = val;
}

bool L1MuDTTFParameters::get_soc_openlut_extr(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return soc_openlut_extr[check(wh,sc)][sc];
}

int L1MuDTTFParameters::check(int wh, int sc) const {

  if ( sc<0 || sc>11 || wh==0 || wh>3 || wh<-3 ) return -99; 

  if ( wh < 0 ) return wh+3;
  else return wh+2;
}
