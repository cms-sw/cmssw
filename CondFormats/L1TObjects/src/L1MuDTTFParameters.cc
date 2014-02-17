//-------------------------------------------------
//
//   Class: L1MuDTTFParameters
//
//   Description: DTTF Parameters from OMDS
//
//
//   $Date: 2009/05/12 10:38:35 $
//   $Revision: 1.5 $
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

#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

using namespace std;

// --------------------------------
//       class L1MuDTTFParameters
//---------------------------------

//--------------
// Operations --
//--------------

//
// reset parameters
//
void L1MuDTTFParameters::reset() {

  for( int i=0; i<6; i++ ) {
    for( int j=0; j<12; j++ ) {

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

void L1MuDTTFParameters::print() const {

  cout << endl;
  cout << "L1 barrel Track Finder Parameters :" << endl;
  cout << "===================================" << endl;
  cout << endl;

  cout << endl;
  cout << "Quality Cut St.1 :" << endl;
  cout << "==================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_qual_st1(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut St.2 :" << endl;
  cout << "==================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_qual_st2(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut St.3 :" << endl;
  cout << "==================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_qual_st3(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut St.4 :" << endl;
  cout << "==================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_qual_st4(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut Next Wheel :" << endl;
  cout << "========================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_stdis_n(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut WL :" << endl;
  cout << "================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_stdis_wl(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut WR :" << endl;
  cout << "================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_stdis_wr(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut ZL :" << endl;
  cout << "================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_stdis_zl(i,j); }
    cout << endl; }

  cout << endl;
  cout << " Quality Cut ZR :" << endl;
  cout << "=================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_stdis_zr(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut SOC St.1 :" << endl;
  cout << "======================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_qcut_st1(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut SOC St.2 :" << endl;
  cout << "======================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_qcut_st2(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Quality Cut SOC St.4 :" << endl;
  cout << "======================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_qcut_st4(i,j); }
    cout << endl; }

  cout << endl;
  cout << "CSC Quality Cut :" << endl;
  cout << "=================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_qual_csc(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Extrapolation 21 :" << endl;
  cout << "==================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_run_21(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Herbert Scheme :" << endl;
  cout << "================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_nbx_del(i,j); }
    cout << endl; }

  cout << endl;
  cout << "CSC Eta Cancellation :" << endl;
  cout << "======================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_csc_etacanc(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Open LUTs :" << endl;
  cout << "===========" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_soc_openlut_extr(i,j); }
    cout << endl; }

}
