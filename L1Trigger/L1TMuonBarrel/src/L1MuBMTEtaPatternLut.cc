//-------------------------------------------------
//
//   Class: L1MuDTEtaPatternLut
//
//   Description: Look-up table for eta track finder
//
//
//   $Date: 2007/03/30 07:48:02 $
//   $Revision: 1.1 $
//
//   Author :
//   N. Neumeister            CERN EP
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTEtaPatternLut.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

// --------------------------------
//       class L1MuDTEtaPatternLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMTEtaPatternLut::L1MuBMTEtaPatternLut() {
  //  if ( load() != 0 ) {
  //    cout << "Can not open files to load eta track finder look-up tables for DTTrackFinder!" << endl;
  //  }

  //  if ( L1MuDTTFConfig::Debug(6) ) print();
}

//--------------
// Destructor --
//--------------

L1MuBMTEtaPatternLut::~L1MuBMTEtaPatternLut() { m_lut.clear(); }

//--------------
// Operations --
//--------------

//
// reset look-up table
//
void L1MuBMTEtaPatternLut::reset() { m_lut.clear(); }

//
// load pattern look-up table for ETF
//
int L1MuBMTEtaPatternLut::load() {
  // get directory name
  string defaultPath = "L1Trigger/";
  string eau_dir = "L1TMuon/data/bmtf_luts/LUTs_Ass/";

  // assemble file name
  edm::FileInPath lut_f = edm::FileInPath(string(defaultPath + eau_dir + "ETFPatternList.lut"));
  string etf_file = lut_f.fullPath();

  // open file
  L1TriggerLutFile file(etf_file);
  if (file.open() != 0)
    return -1;
  //  if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : "
  //                                       << file.getName() << endl;

  // ignore comment lines (always at the beginning)
  int skip2 = getIgnoredLines(file);
  file.ignoreLines(skip2);

  // read patterns
  while (file.good()) {
    int id = file.readInteger();
    if (!file.good())
      break;
    string pat = file.readString();
    if (!file.good())
      break;
    int qual = file.readInteger();
    if (!file.good())
      break;
    int eta = file.readInteger();
    if (!file.good())
      break;
    L1MuDTEtaPattern pattern(id, pat, eta, qual);

    m_lut[pattern.id()] = pattern;

    if (!file.good()) {
      file.close();
      break;
    }
  }

  file.close();

  return 0;
}

//
// print pattern look-up table for ETF
//
void L1MuBMTEtaPatternLut::print() const {
  cout << endl;
  cout << "L1 barrel Track Finder ETA Pattern look-up table :" << endl;
  cout << "==================================================" << endl;
  cout << endl;

  cout << "ETF Patterns : " << m_lut.size() << endl;
  cout << "======================" << endl;
  cout << endl;

  LUT::const_iterator iter = m_lut.begin();
  while (iter != m_lut.end()) {
    cout << (*iter).second << endl;
    iter++;
  }

  cout << endl;
}

//
// get pattern with a given ID
//
L1MuDTEtaPattern L1MuBMTEtaPatternLut::getPattern(int id) const {
  LUT::const_iterator it = m_lut.find(id);
  if (it == m_lut.end()) {
    edm::LogError("L1MuBMTEtaPatternLut: fine eta not found")
        << "Error: L1MuBMTEtaPatternLut: pattern not found : " << id << endl;
  }
  return (*it).second;
}

int L1MuBMTEtaPatternLut::getIgnoredLines(L1TriggerLutFile file) const {
  if (file.open() != 0)
    return -1;
  int skip = 0;
  while (file.good()) {
    string str = file.readString();
    if (str.find('#') == 0)
      skip += 1;
    //cout<<"here "<<str<<" found "<<str.find("#")<<endl;
    if (!file.good()) {
      file.close();
      break;
    }
  }
  file.close();
  return skip;
}
