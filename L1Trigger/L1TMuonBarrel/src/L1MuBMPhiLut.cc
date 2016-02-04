//-------------------------------------------------
//
//   Class: L1MuBMPhiLut
//
//   Description: Look-up tables for phi assignment
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

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMPhiLut.h"

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
//       class L1MuBMPhiLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMPhiLut::L1MuBMPhiLut(const L1TMuonBarrelParams& l1params) {
    l1tbmphiparams = &l1params;
}


//--------------
// Destructor --
//--------------

L1MuBMPhiLut::~L1MuBMPhiLut() {

}


//--------------
// Operations --
//--------------

//
// print phi-assignment look-up tables
//
void L1MuBMPhiLut::print() const {
  unsigned short int nbit_phi  = l1tbmphiparams->get_PHI_Assignment_nbits_Phi();
  unsigned short int nbit_phib = l1tbmphiparams->get_PHI_Assignment_nbits_PhiB();

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
  std::vector<LUT> phi_lut = l1tbmphiparams->phi_lut();

    LUT::const_iterator iter = phi_lut[idx].begin();
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
// get delta-phi value for a given address
//
int L1MuBMPhiLut::getDeltaPhi(int idx, int address) const {
  std::vector<LUT> phi_lut = l1tbmphiparams->phi_lut();
  LUT::const_iterator iter =  phi_lut[idx].find(address);
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
pair<unsigned short, unsigned short> L1MuBMPhiLut::getPrecision() const {

  return pair<unsigned short, unsigned short>(l1tbmphiparams->get_PHI_Assignment_nbits_Phi()
                                              ,l1tbmphiparams->get_PHI_Assignment_nbits_PhiB());

}
