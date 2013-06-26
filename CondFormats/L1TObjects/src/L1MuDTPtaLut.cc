//-------------------------------------------------
//
//   Class: L1MuDTPtaLut
//
//   Description: Look-up tables for pt assignment 
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

#include "CondFormats/L1TObjects/interface/L1MuDTPtaLut.h"

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
#include "CondFormats/L1TObjects/interface/L1MuDTAssParam.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"

using namespace std;

// --------------------------------
//       class L1MuDTPtaLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTPtaLut::L1MuDTPtaLut() : 
                  pta_lut(0), 
                  pta_threshold(MAX_PTASSMETH/2) {

  pta_lut.reserve(MAX_PTASSMETH);
  pta_threshold.reserve(MAX_PTASSMETH/2);
  setPrecision();
  
  //  if ( load() != 0 ) {
  //    cout << "Can not open files to load pt-assignment look-up tables for DTTrackFinder!" << endl;
  //  }

  //  if ( L1MuDTTFConfig::Debug(6) ) print();
  
}


//--------------
// Destructor --
//--------------

L1MuDTPtaLut::~L1MuDTPtaLut() {

  vector<LUT>::iterator iter;
  for ( iter = pta_lut.begin(); iter != pta_lut.end(); iter++ ) {
    (*iter).clear();
  }

  pta_lut.clear();
  pta_threshold.clear();

}


//--------------
// Operations --
//--------------

//
// reset pt-assignment look-up tables
//
void L1MuDTPtaLut::reset() {

  pta_lut.clear();
  pta_threshold.clear();

}


//
// load pt-assignment look-up tables
//
int L1MuDTPtaLut::load() {

  // get directory name
  string defaultPath = "L1TriggerConfig/DTTrackFinder/parameters/";
  string pta_dir = "L1TriggerData/DTTrackFinder/Ass/";
  string pta_str = "";

  // precision : in the look-up tables the following precision is used :
  // phi ...12 bits (address) and  pt ...5 bits
  // now convert phi and phib to the required precision

  int sh_phi  = 12 - nbit_phi;

  // loop over all pt-assignment methods
  for ( int pam = 0; pam < MAX_PTASSMETH; pam++ ) { 
    switch ( pam ) {
      case PT12L  : { pta_str = "pta12l"; break; }
      case PT12H  : { pta_str = "pta12h"; break; }
      case PT13L  : { pta_str = "pta13l"; break; }
      case PT13H  : { pta_str = "pta13h"; break; }
      case PT14L  : { pta_str = "pta14l"; break; }
      case PT14H  : { pta_str = "pta14h"; break; }
      case PT23L  : { pta_str = "pta23l"; break; }
      case PT23H  : { pta_str = "pta23h"; break; }
      case PT24L  : { pta_str = "pta24l"; break; }
      case PT24H  : { pta_str = "pta24h"; break; }
      case PT34L  : { pta_str = "pta34l"; break; }
      case PT34H  : { pta_str = "pta34h"; break; }
      case PT12LO : { pta_str = "pta12l_ovl"; break; }
      case PT12HO : { pta_str = "pta12h_ovl"; break; }
      case PT13LO : { pta_str = "pta13l_ovl"; break; }
      case PT13HO : { pta_str = "pta13h_ovl"; break; }
      case PT14LO : { pta_str = "pta14l_ovl"; break; }
      case PT14HO : { pta_str = "pta14h_ovl"; break; }
      case PT23LO : { pta_str = "pta23l_ovl"; break; }
      case PT23HO : { pta_str = "pta23h_ovl"; break; }
      case PT24LO : { pta_str = "pta24l_ovl"; break; }
      case PT24HO : { pta_str = "pta24h_ovl"; break; }
      case PT34LO : { pta_str = "pta34l_ovl"; break; }
      case PT34HO : { pta_str = "pta34h_ovl"; break; }
      case PT15LO : { pta_str = "pta15l_ovl"; break; }
      case PT15HO : { pta_str = "pta15h_ovl"; break; }
      case PT25LO : { pta_str = "pta25l_ovl"; break; }
      case PT25HO : { pta_str = "pta25h_ovl"; break; }      
    }

    // assemble file name
    edm::FileInPath lut_f = edm::FileInPath(string(defaultPath + pta_dir + pta_str + ".lut"));
    string pta_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(pta_file);
    if ( file.open() != 0 ) return -1;
    //    if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : " 
    //                                         << file.getName() << endl; 

    // get the right shift factor
    int shift = sh_phi;
    int adr_old = -2048 >> shift;

    LUT tmplut;

    int number = -1;
    int sum_pt = 0;

    if ( file.good() ) {
      int threshold = file.readInteger();
      pta_threshold[pam/2] = threshold;
    }
    
    // read values and shift to correct precision
    while ( file.good() ) {
        
      int adr = (file.readInteger()) >> shift;
      int pt  = file.readInteger();

      number++;
      
      if ( adr != adr_old ) {
        assert(number);
        tmplut.insert(make_pair( adr_old, (sum_pt/number) ));

        adr_old = adr;
        number = 0;
        sum_pt = 0;
      }
      
      sum_pt += pt;
      
      if ( !file.good() ) file.close();
      
    }

    file.close();
    pta_lut.push_back(tmplut);
  }
  return 0;

}


//
// print pt-assignment look-up tables
//
void L1MuDTPtaLut::print() const {

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

    LUT::const_iterator iter = pta_lut[pam].begin();
    while ( iter != pta_lut[pam].end() ) {
      int address = (*iter).first;
      int value   = (*iter).second;

      DTTFBitArray<12> b_address(static_cast<unsigned>(abs(address)));
      DTTFBitArray<5> b_value(static_cast<unsigned>(abs(value)));

      if ( address < 0 ) b_address.twoComplement();

      cout.setf(ios::right,ios::adjustfield);
      cout << " " << setbase(10) << setw(5) << address << " (";
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
int L1MuDTPtaLut::getPt(int pta_ind, int address) const {

  LUT::const_iterator iter = pta_lut[pta_ind].find(address);
  if ( iter != pta_lut[pta_ind].end() ) {
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
int L1MuDTPtaLut::getPtLutThreshold(int pta_ind) const {

  if ( pta_ind >= 0 && pta_ind < MAX_PTASSMETH/2 ) {
    return pta_threshold[pta_ind];
  }
  else {
    cerr << "PtaLut::getPtLutThreshold : can not find threshold " << pta_ind << endl;
    return 0;
  }

}


//
// set precision for look-up tables
//
void L1MuDTPtaLut::setPrecision() {

  nbit_phi  = 12;  

}
