//-------------------------------------------------
//
//   Class: L1MuDTPhiLut
//
//   Description: Look-up tables for phi assignment 
//
//
//   $Date: 2012/08/05 12:48:33 $
//   $Revision: 1.8 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "CondFormats/L1TObjects/interface/L1MuDTPhiLut.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cstdlib>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/L1TObjects/interface/DTTFBitArray.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"

using namespace std;

// --------------------------------
//       class L1MuDTPhiLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTPhiLut::L1MuDTPhiLut() {

  phi_lut.reserve(2);
  setPrecision();
  //  if ( load() != 0 ) {
  //    cout << "Can not open files to load phi-assignment look-up tables for DTTrackFinder!" << endl;
  //  }

  //  if ( L1MuDTTFConfig::Debug(6) ) print();

}


//--------------
// Destructor --
//--------------

L1MuDTPhiLut::~L1MuDTPhiLut() {

  vector<LUT>::iterator iter;
  for ( iter = phi_lut.begin(); iter != phi_lut.end(); iter++ ) {
    (*iter).clear();
  }

  phi_lut.clear();

}


//--------------
// Operations --
//--------------

//
// reset phi-assignment look-up tables
//
void L1MuDTPhiLut::reset() {

  phi_lut.clear();

}


//
// load phi-assignment look-up tables
//
int L1MuDTPhiLut::load() {

  // get directory name
  string defaultPath = "L1TriggerConfig/DTTrackFinder/parameters/";
  string phi_dir = "L1TriggerData/DTTrackFinder/Ass/";
  string phi_str = "";

  // precision : in the look-up tables the following precision is used :
  // address (phib) ...10 bits, phi ... 12 bits

  int sh_phi  = 12 - nbit_phi;
  int sh_phib = 10 - nbit_phib;

  // loop over all phi-assignment methods
  for ( int idx = 0; idx < 2; idx++ ) {
    switch ( idx ) {
      case 0 : { phi_str = "phi12"; break; }
      case 1 : { phi_str = "phi42"; break; }
    }

    // assemble file name
    edm::FileInPath lut_f = edm::FileInPath(string(defaultPath + phi_dir + phi_str + ".lut"));
    string phi_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(phi_file);
    if ( file.open() != 0 ) return -1;
    //    if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : " 
    //                                         << file.getName() << endl; 

    LUT tmplut;

    int number = -1;
    int adr_old = -512 >> sh_phib;
    int sum_phi = 0;

    // read values
    while ( file.good() ) {
    
      int adr = (file.readInteger()) >> sh_phib;
      int phi =  file.readInteger();
      
      number++;

      if ( adr != adr_old ) {
        assert(number);
        tmplut.insert(make_pair( adr_old, ((sum_phi/number) >> sh_phi) ));

        adr_old = adr;
        number = 0;
        sum_phi  = 0;
      }
      
      sum_phi += phi;

      if ( !file.good() ) file.close();

    }

    file.close();
    phi_lut.push_back(tmplut);
  } 
  return 0;

}


//
// print phi-assignment look-up tables
//
void L1MuDTPhiLut::print() const {

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

    LUT::const_iterator iter = phi_lut[idx].begin();
    while ( iter != phi_lut[idx].end() ) {
      int address = (*iter).first;
      int value   = (*iter).second;

      DTTFBitArray<10> b_address(static_cast<unsigned>(abs(address)));
      DTTFBitArray<12> b_value(static_cast<unsigned>(abs(value)));

      if ( address < 0 ) b_address.twoComplement();
      if ( value < 0 ) b_value.twoComplement();

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
int L1MuDTPhiLut::getDeltaPhi(int idx, int address) const {

  LUT::const_iterator iter = phi_lut[idx].find(address);
  if ( iter != phi_lut[idx].end() ) {
    return (*iter).second;
  }
  else {
    cerr << "PhiLut::getDeltaPhi : can not find address " << address << endl;
    return 0;
  }

}


//
// set precision for look-up tables
//
void L1MuDTPhiLut::setPrecision() {

  nbit_phi  = 12;
  nbit_phib = 10;

}


//
// get precision for look-up tables
//
pair<unsigned short, unsigned short> L1MuDTPhiLut::getPrecision() const {

  return pair<unsigned short, unsigned short>(nbit_phi,nbit_phib);

}
