//-------------------------------------------------
//
//   Class: L1MuBMPtaLut
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

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMPtaLut.h"

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
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMAssParam.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"


using namespace std;

// --------------------------------
//       class L1MuBMPtaLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMPtaLut::L1MuBMPtaLut(const L1TMuonBarrelParams &l1params) {
    l1tbmparams = &l1params;
}


//--------------
// Destructor --
//--------------

L1MuBMPtaLut::~L1MuBMPtaLut() {
}


//--------------
// Operations --
//--------------

//
// print pt-assignment look-up tables
//
void L1MuBMPtaLut::print() const {
  int nbit_phi = l1tbmparams->get_PT_Assignment_nbits_Phi();

  cout << endl;
  cout << "L1 barrel Track Finder Pt-Assignment look-up tables :" << endl;
  cout << "=====================================================" << endl;
  cout << endl;
  cout << "Precision : " << endl;
  cout << '\t' << setw(2) << nbit_phi  << " bits are used for phi "  << endl;

  // loop over all pt-assignment methods
  for ( int pam = 0; pam < MAX_PTASSMETH; pam++ ) {

    cout << endl;
    cout << "Pt-Assignment Method : " << static_cast<PtAssMethod>(pam) << endl;
    cout << "============================" << endl;
    cout << endl;

    cout << "\t Threshold : " << getPtLutThreshold(pam/2) << endl << endl;

    int maxbits = nbit_phi;

    cout << "      address";
    for ( int i = 0; i < maxbits; i++ ) cout << ' ';
    cout << "  value" << endl;
    for ( int i = 0; i < maxbits; i++ ) cout << '-';
    cout << "-------------------------" << endl;
    std::vector<LUT> pta_lut = l1tbmparams->pta_lut();

    LUT::const_iterator iter = pta_lut[pam].begin();
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
int L1MuBMPtaLut::getPt(int pta_ind, int address) const {

  std::vector<LUT> pta_lut = l1tbmparams->pta_lut();

  LUT::const_iterator iter = pta_lut[pta_ind].find(address);
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
int L1MuBMPtaLut::getPtLutThreshold(int pta_ind) const {
std::vector<int> pta_threshold = l1tbmparams->pta_threshold();
  if ( pta_ind >= 0 && pta_ind < MAX_PTASSMETH/2 ) {
    return pta_threshold[pta_ind];
  }
  else {
    cerr << "PtaLut::getPtLutThreshold : can not find threshold " << pta_ind << endl;
    return 0;
  }

}
