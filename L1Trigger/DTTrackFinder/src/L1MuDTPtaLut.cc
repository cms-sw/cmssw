//-------------------------------------------------
//
//   Class: L1MuDTPtaLut
//
//   Description: Look-up tables for pt assignment 
//
//
//   $Date: 2006/06/26 16:11:13 $
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

#include "L1Trigger/DTTrackFinder/src/L1MuDTPtaLut.h"

//---------------
// C++ Headers --
//---------------

#include <ostream>
#include <iomanip>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "L1Trigger/DTTrackFinder/interface/BitArray.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTAssignmentUnit.h"
#include "DataFormats/L1DTTrackFinder/interface/L1TriggerLutFile.h"

// --------------------------------
//       class L1MuDTPtaLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTPtaLut::L1MuDTPtaLut() : 
                  pta_lut(0), 
                  pta_threshold(L1MuDTAssignmentUnit::MAX_PTASSMETH/2) {

  pta_lut.reserve(L1MuDTAssignmentUnit::MAX_PTASSMETH);
  pta_threshold.reserve(L1MuDTAssignmentUnit::MAX_PTASSMETH/2);
  setPrecision();
  
  if ( load() != 0 ) {
    cout << "Can not open files to load pt-assignment look-up tables for DTTrackFinder!" << endl;
  }

  if ( L1MuDTTFConfig::Debug(6) ) print();
  
}


//--------------
// Destructor --
//--------------

L1MuDTPtaLut::~L1MuDTPtaLut() {

  vector<LUT*>::iterator iter;
  for ( iter = pta_lut.begin(); iter != pta_lut.end(); iter++ ) {
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
  string defaultPath(getenv("DTTF_DATA_PATH"));
  string pta_dir = "L1TriggerData/DTTrackFinder/Ass/";
  string pta_str = "";

  // precision : in the look-up tables the following precision is used :
  // phi ...12 bits (address) and  pt ...5 bits
  // now convert phi and phib to the required precision

  int sh_phi  = 12 - nbit_phi;

  // loop over all pt-assignment methods
  for ( int pam = 0; pam < L1MuDTAssignmentUnit::MAX_PTASSMETH; pam++ ) { 
    switch ( pam ) {
      case L1MuDTAssignmentUnit::PT12L  : { pta_str = "pta12l"; break; }
      case L1MuDTAssignmentUnit::PT12H  : { pta_str = "pta12h"; break; }
      case L1MuDTAssignmentUnit::PT13L  : { pta_str = "pta13l"; break; }
      case L1MuDTAssignmentUnit::PT13H  : { pta_str = "pta13h"; break; }
      case L1MuDTAssignmentUnit::PT14L  : { pta_str = "pta14l"; break; }
      case L1MuDTAssignmentUnit::PT14H  : { pta_str = "pta14h"; break; }
      case L1MuDTAssignmentUnit::PT23L  : { pta_str = "pta23l"; break; }
      case L1MuDTAssignmentUnit::PT23H  : { pta_str = "pta23h"; break; }
      case L1MuDTAssignmentUnit::PT24L  : { pta_str = "pta24l"; break; }
      case L1MuDTAssignmentUnit::PT24H  : { pta_str = "pta24h"; break; }
      case L1MuDTAssignmentUnit::PT34L  : { pta_str = "pta34l"; break; }
      case L1MuDTAssignmentUnit::PT34H  : { pta_str = "pta34h"; break; }
      case L1MuDTAssignmentUnit::PT12LO : { pta_str = "pta12l_ovl"; break; }
      case L1MuDTAssignmentUnit::PT12HO : { pta_str = "pta12h_ovl"; break; }
      case L1MuDTAssignmentUnit::PT13LO : { pta_str = "pta13l_ovl"; break; }
      case L1MuDTAssignmentUnit::PT13HO : { pta_str = "pta13h_ovl"; break; }
      case L1MuDTAssignmentUnit::PT14LO : { pta_str = "pta14l_ovl"; break; }
      case L1MuDTAssignmentUnit::PT14HO : { pta_str = "pta14h_ovl"; break; }
      case L1MuDTAssignmentUnit::PT23LO : { pta_str = "pta23l_ovl"; break; }
      case L1MuDTAssignmentUnit::PT23HO : { pta_str = "pta23h_ovl"; break; }
      case L1MuDTAssignmentUnit::PT24LO : { pta_str = "pta24l_ovl"; break; }
      case L1MuDTAssignmentUnit::PT24HO : { pta_str = "pta24h_ovl"; break; }
      case L1MuDTAssignmentUnit::PT34LO : { pta_str = "pta34l_ovl"; break; }
      case L1MuDTAssignmentUnit::PT34HO : { pta_str = "pta34h_ovl"; break; }
      case L1MuDTAssignmentUnit::PT15LO : { pta_str = "pta15l_ovl"; break; }
      case L1MuDTAssignmentUnit::PT15HO : { pta_str = "pta15h_ovl"; break; }
      case L1MuDTAssignmentUnit::PT25LO : { pta_str = "pta25l_ovl"; break; }
      case L1MuDTAssignmentUnit::PT25HO : { pta_str = "pta25h_ovl"; break; }      
    }

    // assemble file name
    edm::FileInPath lut_f = edm::FileInPath(string("L1Trigger/DTTrackFinder/parameters/" + pta_dir + pta_str + ".lut"));
    string pta_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(pta_file);
    if ( file.open() != 0 ) return -1;
    if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : " 
                                         << file.getName() << endl; 

    // get the right shift factor
    int shift = sh_phi;
    int adr_old = -2048 >> shift;

    LUT* tmplut = new LUT;
    pta_lut.push_back(tmplut);

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
        tmplut->insert(make_pair( adr_old, (sum_pt/number) ));

        adr_old = adr;
        number = 0;
        sum_pt = 0;
      }
      
      sum_pt += pt;
      
      if ( !file.good() ) file.close();
      
    }

    file.close();
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
  for ( int pam = 0; pam < L1MuDTAssignmentUnit::MAX_PTASSMETH; pam++ ) {
 
    cout << endl;
    cout << "Pt-Assignment Method : " << static_cast<L1MuDTAssignmentUnit::PtAssMethod>(pam) << endl;
    cout << "============================" << endl;
    cout << endl;

    cout << "\t Threshold : " << getPtLutThreshold(pam/2) << endl << endl;

    int maxbits = nbit_phi;
  
    cout << "      address";
    for ( int i = 0; i < maxbits; i++ ) cout << ' ';
    cout << "  value" << endl;
    for ( int i = 0; i < maxbits; i++ ) cout << '-';    
    cout << "-------------------------" << endl;

    LUT::const_iterator iter = pta_lut[pam]->begin();
    while ( iter != pta_lut[pam]->end() ) {
      int address = (*iter).first;
      int value   = (*iter).second;

      BitArray<12> b_address(static_cast<unsigned>(abs(address)));
      BitArray<5> b_value(static_cast<unsigned>(abs(value)));

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

  LUT::const_iterator iter = pta_lut[pta_ind]->find(address);
  if ( iter != pta_lut[pta_ind]->end() ) {
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

  if ( pta_ind >= 0 && pta_ind < L1MuDTAssignmentUnit::MAX_PTASSMETH/2 ) {
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
  nbit_phi = L1MuDTTFConfig::getNbitsPtaPhi();

}
