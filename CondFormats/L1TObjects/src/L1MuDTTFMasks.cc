//-------------------------------------------------
//
//   Class: L1MuDTTFMasks
//
//   Description: DTTF Masks from OMDS
//
//
//   $Date: 2009/07/22 07:27:50 $
//   $Revision: 1.3 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"

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
//       class L1MuDTTFMasks
//---------------------------------

//--------------
// Operations --
//--------------

//
// reset parameters
//
void L1MuDTTFMasks::reset() {

  for( int i=0; i<6; i++ ) {
    for( int j=0; j<12; j++ ) {

      inrec_chdis_st1[i][j]  = false;
      inrec_chdis_st2[i][j]  = false;
      inrec_chdis_st3[i][j]  = false;
      inrec_chdis_st4[i][j]  = false;
      inrec_chdis_csc[i][j]  = false;
      etsoc_chdis_st1[i][j]  = false;
      etsoc_chdis_st2[i][j]  = false;
      etsoc_chdis_st3[i][j]  = false;
    }
  }
}

void L1MuDTTFMasks::set_inrec_chdis_st1(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_st1[check(wh,sc)][sc] = val;
}

bool L1MuDTTFMasks::get_inrec_chdis_st1(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_st1[check(wh,sc)][sc];
}

void L1MuDTTFMasks::set_inrec_chdis_st2(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_st2[check(wh,sc)][sc] = val;
}

bool L1MuDTTFMasks::get_inrec_chdis_st2(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_st2[check(wh,sc)][sc];
}

void L1MuDTTFMasks::set_inrec_chdis_st3(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_st3[check(wh,sc)][sc] = val;
}

bool L1MuDTTFMasks::get_inrec_chdis_st3(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_st3[check(wh,sc)][sc];
}

void L1MuDTTFMasks::set_inrec_chdis_st4(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_st4[check(wh,sc)][sc] = val;
}

bool L1MuDTTFMasks::get_inrec_chdis_st4(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_st4[check(wh,sc)][sc];
}

void L1MuDTTFMasks::set_inrec_chdis_csc(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  inrec_chdis_csc[check(wh,sc)][sc] = val;
}

bool L1MuDTTFMasks::get_inrec_chdis_csc(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return inrec_chdis_csc[check(wh,sc)][sc];
}

void L1MuDTTFMasks::set_etsoc_chdis_st1(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  etsoc_chdis_st1[check(wh,sc)][sc] = val;
}

bool L1MuDTTFMasks::get_etsoc_chdis_st1(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return etsoc_chdis_st1[check(wh,sc)][sc];
}

void L1MuDTTFMasks::set_etsoc_chdis_st2(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  etsoc_chdis_st2[check(wh,sc)][sc] = val;
}

bool L1MuDTTFMasks::get_etsoc_chdis_st2(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return etsoc_chdis_st2[check(wh,sc)][sc];
}

void L1MuDTTFMasks::set_etsoc_chdis_st3(int wh, int sc, const bool val) {

  if ( check(wh,sc) == -99 ) return;
  etsoc_chdis_st3[check(wh,sc)][sc] = val;
}

bool L1MuDTTFMasks::get_etsoc_chdis_st3(int wh, int sc) const{

  if ( check(wh,sc) == -99 ) return false;
  return etsoc_chdis_st3[check(wh,sc)][sc];
}

int L1MuDTTFMasks::check(int wh, int sc) const {

  if ( sc<0 || sc>11 || wh==0 || wh>3 || wh<-3 ) return -99; 

  if ( wh < 0 ) return wh+3;
  else return wh+2;
}

void L1MuDTTFMasks::print() const {

  cout << endl;
  cout << "L1 barrel Track Finder Masks :" << endl;
  cout << "==============================" << endl;
  cout << endl;

  cout << endl;
  cout << "Disable PHTF St.1 :" << endl;
  cout << "===================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_chdis_st1(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Disable PHTF St.2 :" << endl;
  cout << "===================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_chdis_st2(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Disable PHTF St.3 :" << endl;
  cout << "===================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_chdis_st3(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Disable PHTF St.4 :" << endl;
  cout << "===================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_chdis_st4(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Disable CSC :" << endl;
  cout << "=============" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_inrec_chdis_csc(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Disable ETTF St.1 :" << endl;
  cout << "===================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_etsoc_chdis_st1(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Disable ETTF St.2 :" << endl;
  cout << "===================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_etsoc_chdis_st2(i,j); }
    cout << endl; }

  cout << endl;
  cout << "Disable ETTF St.3 :" << endl;
  cout << "===================" << endl;
  cout << endl;
  for( int i=-3; i<4; i++ ) {
    if ( i == 0 ) continue;
    for( int j=0; j<12; j++ ) { cout << " " << setw(1) << get_etsoc_chdis_st3(i,j); }
    cout << endl; }

}
