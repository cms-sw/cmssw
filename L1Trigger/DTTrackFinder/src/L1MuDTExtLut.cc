//-------------------------------------------------
//
//   Class: L1MuDTExtLut
//
//   Description: Look-up tables for extrapolation
//
//
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------
using namespace std;

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTExtLut.h"

//---------------
// C++ Headers --
//---------------

#include <ostream>
#include <iomanip>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/interface/BitArray.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTExtParam.h"
#include "DataFormats/L1DTTrackFinder/interface/L1TriggerLutFile.h"

// --------------------------------
//       class L1MuDTExtLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTExtLut::L1MuDTExtLut() {

  ext_lut.reserve(MAX_EXT);
  setPrecision();
  if ( load() != 0 ) {
    cout << "Can not open files to load  extrapolation look-up tables for DTTrackFinder!" << endl;
  }

  print();

}


//--------------
// Destructor --
//--------------

L1MuDTExtLut::~L1MuDTExtLut() {

  typedef vector<LUT*>::iterator LI;
  for ( LI iter = ext_lut.begin(); iter != ext_lut.end(); iter++ ) {
    if (*iter != 0 ) {
      delete *iter;
      *iter = 0;
    }
  }

}


//--------------
// Operations --
//--------------

//
// reset extrapolation look-up tables
//
void L1MuDTExtLut::reset() {

  ext_lut.clear();

}


//
// load extrapolation look-up tables
//
int L1MuDTExtLut::load() {

  // get directory name
  string defaultPath(getenv("DTTF_DATA_PATH"));
  string ext_dir = "L1TriggerData/DTTrackFinder/Ext/";
  string ext_str = "";

  // precision : in the look-up tables the following precision is used :
  // phi ...12 bits (low, high), phib ...10 bits (address)
  // now convert phi and phib to the required precision

  int sh_phi  = 12 - nbit_phi;
  int sh_phib = 10 - nbit_phib;

  // loop over all extrapolations
  for ( int ext = 0; ext < MAX_EXT; ext++ ) { 
    switch (ext) {
      case EX12 : ext_str = "ext12"; break;
      case EX13 : ext_str = "ext13"; break;
      case EX14 : ext_str = "ext14"; break;
      case EX21 : ext_str = "ext21"; break;
      case EX23 : ext_str = "ext23"; break;
      case EX24 : ext_str = "ext24"; break;
      case EX34 : ext_str = "ext34"; break;
      case EX15 : ext_str = "ext15"; break;
      case EX16 : ext_str = "ext16"; break;
      case EX25 : ext_str = "ext25"; break;
      case EX26 : ext_str = "ext26"; break;
      case EX56 : ext_str = "ext56"; break;
    }

    // assemble file name
    string ext_file = ext_dir + ext_str + ".lut";

    // open file
    L1TriggerLutFile file(defaultPath+ext_file);
    if ( file.open() != 0 ) return -1;
    if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : " 
                                         << file.getName() << endl; 

    LUT* tmplut = new LUT;
    ext_lut.push_back(tmplut);

    int number = -1;
    int adr_old = -512 >> sh_phib;
    int sum_low = 0;
    int sum_high = 0;

    // read values and shift to correct precision
    while ( file.good() ) {
 
      int adr  = ( file.readInteger() ) >> sh_phib;	// address (phib)
      int low  = ( file.readInteger() );    		// low value (phi)
      int high = ( file.readInteger() );	        // high value (phi)

      number++;
      
      if ( adr != adr_old ) {
      
        tmplut->low[adr_old]  = sum_low  >> sh_phi;
        tmplut->high[adr_old] = sum_high >> sh_phi;

        adr_old = adr;
        number = 0;
        sum_low  = 0;
        sum_high = 0;

      }

      if (number == 0) sum_low  = low;
      if (number == 0) sum_high = high;

      if ( !file.good() ) file.close();
    }

    file.close();
    
  }
  return 0;

}


//
// print extrapolation look-up tables
//
void L1MuDTExtLut::print() const {

  cout << endl;
  cout << "L1 barrel Track Finder Extrapolation look-up tables :" << endl;
  cout << "=====================================================" << endl;
  cout << endl;
  cout << "Precision : " << endl;
  cout << '\t' << setw(2) << nbit_phi  << " bits are used for phi "  << endl;
  cout << '\t' << setw(2) << nbit_phib << " bits are used for phib " << endl;

  // loop over all extrapolations
  for ( int ext = 0; ext < MAX_EXT; ext++ ) {

    cout << endl;
    cout << "Extrapolation : " << static_cast<Extrapolation>(ext) << endl;
    cout << "====================" << endl;
    cout << endl;

    cout << "      address";
    for ( int i = 0; i < nbit_phib; i++ ) cout << ' ';
    cout << "    low-value";
    for ( int i = 0; i < nbit_phi; i++ ) cout << ' ';
    cout << "  high-value      " << endl;
    for ( int i = 0; i < 2*nbit_phi + nbit_phib; i++ ) cout << '-';
    cout << "---------------------------------" << endl;

    LUT::LUTmap::const_iterator iter = ext_lut[ext]->low.begin();
    LUT::LUTmap::const_iterator iter1;
    while ( iter != ext_lut[ext]->low.end() ) {
      int address = (*iter).first;
      int low     = (*iter).second;
      iter1 = ext_lut[ext]->high.find(address);
      int high    = (*iter1).second;

      BitArray<10> b_address(static_cast<unsigned>(abs(address)));
      BitArray<12> b_low(static_cast<unsigned>(abs(low)));
      BitArray<12> b_high(static_cast<unsigned>(abs(high)));

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


//
// get low_value for a given address
//
int L1MuDTExtLut::getLow(int ext_ind, int address) const {

  LUT::LUTmap::const_iterator iter = ext_lut[ext_ind]->low.find(address);
  if ( iter != ext_lut[ext_ind]->low.end() ) {
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
int L1MuDTExtLut::getHigh(int ext_ind, int address) const {

  LUT::LUTmap::const_iterator iter = ext_lut[ext_ind]->high.find(address);
  if ( iter != ext_lut[ext_ind]->high.end() ) {
    return (*iter).second;
  }
  else {
    cerr << "ExtLut::getHigh : can not find address " << address << endl;
    return 99999;
  }
}


//
// set precision for Look-up tables
//
void L1MuDTExtLut::setPrecision() {

  nbit_phi  = 12;
  nbit_phib = 10;
  nbit_phi  = L1MuDTTFConfig::getNbitsExtPhi();
  nbit_phib = L1MuDTTFConfig::getNbitsExtPhib();

}
