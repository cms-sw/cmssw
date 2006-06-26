//-------------------------------------------------
//
//   Class: L1MuDTEtaPatternLut
//
//   Description: Look-up table for eta track finder
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

#include "L1Trigger/DTTrackFinder/src/L1MuDTEtaPatternLut.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "DataFormats/L1DTTrackFinder/interface/L1TriggerLutFile.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTEtaPattern.h"

// --------------------------------
//       class L1MuDTEtaPatternLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTEtaPatternLut::L1MuDTEtaPatternLut() {

  if ( load() != 0 ) {
    cout << "Can not open files to load eta track finder look-up tables for DTTrackFinder!" << endl;
  }

  if ( L1MuDTTFConfig::Debug(6) ) print();

}


//--------------
// Destructor --
//--------------

L1MuDTEtaPatternLut::~L1MuDTEtaPatternLut() {

  ETFLut_iter iter = m_lut.begin();
  while ( iter != m_lut.end() ) {
    if ( (*iter).second != 0 ) { 
      delete (*iter).second;
      (*iter).second = 0;
    }  
    iter++;
  }
  m_lut.clear();

}


//--------------
// Operations --
//--------------

//
// reset look-up table
//
void L1MuDTEtaPatternLut::reset() {

  m_lut.clear();

}


//
// load pattern look-up table for ETF
//
int L1MuDTEtaPatternLut::load() {

  // get directory name
  string defaultPath(getenv("DTTF_DATA_PATH"));
  string eau_dir = "L1TriggerData/DTTrackFinder/Eau/";

  // assemble file name
  string etf_file = eau_dir + "ETFPatternList.lut";
   
  // open file
  L1TriggerLutFile file(defaultPath+etf_file);
  if ( file.open() != 0 ) return -1;
  if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : " 
                                       << file.getName() << endl; 

  // ignore comment lines 
  file.ignoreLines(16);
 
  // read patterns
  while ( file.good() ) {

    int id     = file.readInteger();
    if ( !file.good() ) break;
    string pat = file.readString();
    if ( !file.good() ) break;
    int qual   = file.readInteger();
    if ( !file.good() ) break;
    int eta    = file.readInteger();
    if ( !file.good() ) break;
    L1MuDTEtaPattern* pattern = new L1MuDTEtaPattern(id,pat,eta,qual);
      
    m_lut[pattern->id()] = pattern;

    if ( !file.good() ) { file.close(); break; }
    
  }

  file.close();
    
  return 0;

}


//
// print pattern look-up table for ETF
//
void L1MuDTEtaPatternLut::print() const {

  cout << endl;
  cout << "L1 barrel Track Finder ETA Pattern look-up table :" << endl;
  cout << "==================================================" << endl;
  cout << endl;

  cout << "ETF Patterns : " <<  m_lut.size() << endl;
  cout << "======================" << endl;
  cout << endl;

  LUT::const_iterator iter = m_lut.begin();
  while ( iter != m_lut.end() ) {
    cout << *(*iter).second << endl;
    iter++;
  }

  cout << endl;

}


//
// get pattern with a given ID
//
L1MuDTEtaPattern* L1MuDTEtaPatternLut::getPattern(int id) const {

  LUT::const_iterator it = m_lut.find(id);
  if ( it == m_lut.end() ) {
    cerr << "Error: L1MuDTEtaPatternLut: pattern not found : " << id << endl;
    return 0;
  }
  return (*it).second;  

}
