//-------------------------------------------------
//
//   Class: L1MuBMLUTHandler
//
//   Description: Look-up tables for pt assignment
//
//
//   $Date: 2010/05/12 23:03:43 $
//   $Revision: 1.7 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//   G. Flouris               U. Ioannina
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMLUTHandler.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <stdlib.h> /* getenv */

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/L1TObjects/interface/DTTFBitArray.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"


using namespace std;

// --------------------------------
//       class L1MuBMLUTHandler
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMLUTHandler::L1MuBMLUTHandler(const L1TMuonBarrelParams &l1params) {
    l1tbmparams = &l1params;
}


//--------------
// Destructor --
//--------------

L1MuBMLUTHandler::~L1MuBMLUTHandler() {
}


//--------------
// Operations --
//--------------

//
// print pt-assignment look-up tables
//
void L1MuBMLUTHandler::print_pta_lut() const {
  int nbit_phi = l1tbmparams->get_PT_Assignment_nbits_Phi();

  cout << endl;
  cout << "L1 barrel Track Finder Pt-Assignment look-up tables :" << endl;
  cout << "=====================================================" << endl;
  cout << endl;
  cout << "Precision : " << endl;
  cout << '\t' << setw(2) << nbit_phi  << " bits are used for phi "  << endl;

  // loop over all pt-assignment methods
  for ( int pam = 0; pam < L1MuBMLUTHandler::MAX_PTASSMETH; pam++ ) {

    cout << endl;
    cout << "Pt-Assignment Method : " << static_cast<L1MuBMLUTHandler::PtAssMethod>(pam) << endl;
    cout << "============================" << endl;
    cout << endl;

    cout << "\t Threshold : " << getPtLutThreshold(pam/2) << endl << endl;

    int maxbits = nbit_phi;
    if (pam >= MAX_PTASSMETHA) maxbits = nbit_phi-2;

    cout << "      address";
    for ( int i = 0; i < maxbits; i++ ) cout << ' ';
    cout << "  value" << endl;
    for ( int i = 0; i < maxbits; i++ ) cout << '-';
    cout << "-------------------------" << endl;
    std::vector<L1TMuonBarrelParams::LUT> pta_lut = l1tbmparams->pta_lut();

    L1TMuonBarrelParams::LUT::const_iterator iter = pta_lut[pam].begin();
    while ( iter != pta_lut[pam].end() ) {
      int address = (*iter).first;
      int value   = (*iter).second;

      DTTFBitArray<12> b_address(static_cast<unsigned>(abs(address)));
      DTTFBitArray<9> b_value(static_cast<unsigned>(abs(value)));

      if ( address < 0 ) b_address.twoComplement();

      cout.setf(ios::right,ios::adjustfield);
      cout << " " << setbase(10) << setw(9) << address << " (";
      for ( int i = maxbits-1; i >= 0; i-- ) cout << b_address[i];
      cout << ")   " << setw(3) << value << " (";
      b_value.print();
      cout << ")" << endl;

      iter++;
    }

  }

  cout << endl;

}


//
// get pt value for a given address
//
int L1MuBMLUTHandler::getPt(int pta_ind, int address) const {

  std::vector<L1TMuonBarrelParams::LUT> pta_lut = l1tbmparams->pta_lut();

  L1TMuonBarrelParams::LUT::const_iterator iter = pta_lut[pta_ind].find(address);
  if ( iter != pta_lut[pta_ind].end() ) {
    //std::cout<<"pta_ind  "<<pta_ind<<"  address  "<<address<<"  pt  "<<(*iter).second<<std::endl;

    return (*iter).second;
  }
  else {
    cerr << "PtaLut::getPt : can not find address " << address << endl;
    return 0;
  }

}


//
// get pt-assignment LUT threshold
//
int L1MuBMLUTHandler::getPtLutThreshold(int pta_ind) const {
std::vector<int> pta_threshold = l1tbmparams->pta_threshold();
  if ( pta_ind >= 0 && pta_ind < L1MuBMLUTHandler::MAX_PTASSMETH /2 ) {
    return pta_threshold[pta_ind];
  }
  else {
    cerr << "PtaLut::getPtLutThreshold : can not find threshold " << pta_ind << endl;
    return 0;
  }

}

//
// get delta-phi value for a given address
//
int L1MuBMLUTHandler::getDeltaPhi(int idx, int address) const {
  std::vector<L1TMuonBarrelParams::LUT> phi_lut = l1tbmparams->phi_lut();
  L1TMuonBarrelParams::LUT::const_iterator iter =  phi_lut[idx].find(address);
  if ( iter != phi_lut[idx].end() ) {
    return (*iter).second;
  }
  else {
    cerr << "PhiLut::getDeltaPhi : can not find address " << address << endl;
    return 0;
  }

}

//
// get precision for look-up tables
//
pair<unsigned short, unsigned short> L1MuBMLUTHandler::getPrecision() const {

  return pair<unsigned short, unsigned short>(l1tbmparams->get_PHI_Assignment_nbits_Phi()
                                              ,l1tbmparams->get_PHI_Assignment_nbits_PhiB());

}




// print phi-assignment look-up tables
//
void L1MuBMLUTHandler::print_phi_lut() const {
  unsigned short int nbit_phi  = l1tbmparams->get_PHI_Assignment_nbits_Phi();
  unsigned short int nbit_phib = l1tbmparams->get_PHI_Assignment_nbits_PhiB();

  cout << endl;
  cout << "L1 barrel Track Finder Phi-Assignment look-up tables :" << endl;
  cout << "======================================================" << endl;
  cout << endl;
  cout << "Precision : " << endl;
  cout << '\t' << setw(2) << nbit_phi  << " bits are used for phi "  << endl;
  cout << '\t' << setw(2) << nbit_phib << " bits are used for phib " << endl;

  // loop over all phi-assignment methods
  for ( int idx = 0; idx < 2; idx++ ) {

    cout << endl;
    if ( idx == 0 ) cout << "Phi-Assignment Method : " << "PHI12" << endl;
    if ( idx == 1 ) cout << "Phi-Assignment Method : " << "PHI42" << endl;
    cout << "=============================" << endl;
    cout << endl;

    cout << "      address";
    for ( int i = 0; i < nbit_phib; i++ ) cout << ' ';
    cout << "    value" << endl;
    for ( int i = 0; i < nbit_phi + nbit_phib; i++ ) cout << '-';
    cout << "----------------------" << endl;
  std::vector<L1TMuonBarrelParams::LUT> phi_lut = l1tbmparams->phi_lut();

    L1TMuonBarrelParams::LUT::const_iterator iter = phi_lut[idx].begin();
    while ( iter != phi_lut[idx].end() ) {
      int address = (*iter).first;
      int value   = (*iter).second;

      DTTFBitArray<10> b_address(static_cast<unsigned>(abs(address)));
      DTTFBitArray<12> b_value(static_cast<unsigned>(abs(value)));

      if ( address < 0 ) b_address.twoComplement();
      if ( value < 0 )   b_value.twoComplement();

      cout.setf(ios::right,ios::adjustfield);
      cout << " " << setbase(10) << setw(5) << address << " (";
      for ( int i = nbit_phib-1; i >= 0; i-- ) cout << b_address[i];
      cout << ")   " << setw(5) << value  << " (";
      for ( int i = nbit_phi-1; i >= 0; i-- ) cout << b_value[i];
      cout << ")  " << endl;

      iter++;
    }

  }

  cout << endl;

}



//
// get low_value for a given address
//
int L1MuBMLUTHandler::getLow(int ext_ind, int address) const {

  std::vector<L1TMuonBarrelParams::LUTParams::extLUT> ext_lut = l1tbmparams->ext_lut();
  L1TMuonBarrelParams::LUT::const_iterator iter = ext_lut[ext_ind].low.find(address);
  if ( iter != ext_lut[ext_ind].low.end() ) {
    return (*iter).second;
  }
  else {
    cerr << "ExtLut::getLow : can not find address " << address << endl;
    return 99999;
  }
}


//
// get high_value for a given address
//
int L1MuBMLUTHandler::getHigh(int ext_ind, int address) const {

  std::vector<L1TMuonBarrelParams::LUTParams::extLUT> ext_lut = l1tbmparams->ext_lut();
  L1TMuonBarrelParams::LUT::const_iterator iter = ext_lut[ext_ind].high.find(address);
  if ( iter != ext_lut[ext_ind].high.end() ) {
    return (*iter).second;
  }
  else {
    cerr << "ExtLut::getHigh : can not find address " << address << endl;
    return 99999;
  }
}


//
// print extrapolation look-up tables
//
void L1MuBMLUTHandler::print_ext_lut() const {
  unsigned short int nbit_phi  = l1tbmparams->get_PHI_Assignment_nbits_Phi();
  unsigned short int nbit_phib = l1tbmparams->get_PHI_Assignment_nbits_PhiB();
  cout << endl;
  cout << "L1 barrel Track Finder Extrapolation look-up tables :" << endl;
  cout << "=====================================================" << endl;
  cout << endl;
  cout << "Precision : " << endl;
  cout << '\t' << setw(2) << nbit_phi  << " bits are used for phi "  << endl;
  cout << '\t' << setw(2) << nbit_phib << " bits are used for phib " << endl;

  // loop over all extrapolations
  for ( int ext = 0; ext < L1MuBMLUTHandler::MAX_EXT; ext++ ) {

    cout << endl;
    cout << "Extrapolation : " << static_cast<L1MuBMLUTHandler::Extrapolation>(ext) << endl;
    cout << "====================" << endl;
    cout << endl;

    cout << "      address";
    for ( int i = 0; i < nbit_phib; i++ ) cout << ' ';
    cout << "    low-value";
    for ( int i = 0; i < nbit_phi; i++ ) cout << ' ';
    cout << "  high-value      " << endl;
    for ( int i = 0; i < 2*nbit_phi + nbit_phib; i++ ) cout << '-';
    cout << "---------------------------------" << endl;
    std::vector<L1TMuonBarrelParams::LUTParams::extLUT> ext_lut = l1tbmparams->ext_lut();
    L1TMuonBarrelParams::LUT::const_iterator iter = ext_lut[ext].low.begin();
    L1TMuonBarrelParams::LUT::const_iterator iter1;
    while ( iter != ext_lut[ext].low.end() ) {
      int address = (*iter).first;
      int low     = (*iter).second;
      iter1 = ext_lut[ext].high.find(address);
      int high    = (*iter1).second;

      DTTFBitArray<10> b_address(static_cast<unsigned>(abs(address)));
      DTTFBitArray<12> b_low(static_cast<unsigned>(abs(low)));
      DTTFBitArray<12> b_high(static_cast<unsigned>(abs(high)));

      if ( address < 0 ) b_address.twoComplement();
      if ( low < 0 ) b_low.twoComplement();
      if ( high < 0 ) b_high.twoComplement();

      cout.setf(ios::right,ios::adjustfield);
      cout << " " << setbase(10) << setw(5) << address << " (";
      for ( int i = nbit_phib-1; i >= 0; i-- ) cout << b_address[i];
      cout << ")   " << setw(5) << low  << " (";
      for ( int i = nbit_phi-1; i >= 0; i-- ) cout << b_low[i];
      cout << ")   " << setw(5) << high << " (";
      for ( int i = nbit_phi-1; i >= 0; i-- ) cout << b_high[i];
      cout << ")  " << endl;

      iter++;
    }

  }

  cout << endl;

}

