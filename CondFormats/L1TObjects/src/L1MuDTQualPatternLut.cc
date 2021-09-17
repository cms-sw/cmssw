//-------------------------------------------------
//
//   Class: L1MuDTQualPatternLut
//
//   Description: Look-up tables for eta matching unit (EMU)
//                stores lists of qualified patterns and
//                coarse eta values
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

#include "CondFormats/L1TObjects/interface/L1MuDTQualPatternLut.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Utilities/interface/FileInPath.h"
#include "CondFormats/L1TObjects/interface/L1TriggerLutFile.h"

using namespace std;

// --------------------------------
//       class L1MuDTQualPatternLut
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTQualPatternLut::L1MuDTQualPatternLut() {
  //  if ( load() != 0 ) {
  //    cout << "Can not open files to load eta matching look-up tables for DTTrackFinder!" << endl;
  //  }

  //  if ( L1MuDTTFConfig::Debug(6) ) print();
}

//--------------
// Destructor --
//--------------

L1MuDTQualPatternLut::~L1MuDTQualPatternLut() {
  LUT::iterator iter = m_lut.begin();
  while (iter != m_lut.end()) {
    (*iter).second.second.clear();
    iter++;
  }

  m_lut.clear();
}

//--------------
// Operations --
//--------------

//
// reset look-up tables
//
void L1MuDTQualPatternLut::reset() { m_lut.clear(); }

//
// load look-up tables for EMU
//
int L1MuDTQualPatternLut::load() {
  // get directory name
  string defaultPath = "L1TriggerConfig/DTTrackFinder/parameters/";
  string eau_dir = "L1TriggerData/DTTrackFinder/Eau/";
  string emu_str = "";

  // loop over all sector processors
  for (int sp = 0; sp < 6; sp++) {
    switch (sp) {
      case 0: {
        emu_str = "QualPatternList_SP1";
        break;
      }
      case 1: {
        emu_str = "QualPatternList_SP2";
        break;
      }
      case 2: {
        emu_str = "QualPatternList_SP3";
        break;
      }
      case 3: {
        emu_str = "QualPatternList_SP4";
        break;
      }
      case 4: {
        emu_str = "QualPatternList_SP5";
        break;
      }
      case 5: {
        emu_str = "QualPatternList_SP6";
        break;
      }
    }

    // assemble file name
    edm::FileInPath lut_f = edm::FileInPath(string(defaultPath + eau_dir + emu_str + ".lut"));
    string emu_file = lut_f.fullPath();

    // open file
    L1TriggerLutFile file(emu_file);
    if (file.open() != 0)
      return -1;
    //    if ( L1MuDTTFConfig::Debug(1) ) cout << "Reading file : "
    //                                         << file.getName() << endl;

    // ignore comment lines
    file.ignoreLines(14);

    // read file
    while (file.good()) {
      int id = file.readInteger();
      if (!file.good())
        break;
      int eta = file.readInteger();
      if (!file.good())
        break;
      int num = file.readInteger();
      if (!file.good())
        break;

      vector<short> patternlist;
      for (int i = 0; i < num; i++) {
        int pattern = file.readInteger();
        patternlist.push_back(pattern);
      }

      m_lut[make_pair(sp + 1, id)] = make_pair(eta, patternlist);

      if (!file.good()) {
        file.close();
        break;
      }
    }

    file.close();
  }

  return 0;
}

//
// print look-up tables for EMU
//
void L1MuDTQualPatternLut::print() const {
  cout << endl;
  cout << "L1 barrel Track Finder Qual Pattern look-up tables :" << endl;
  cout << "====================================================" << endl;
  cout << endl;

  int spold = 0;

  LUT::const_iterator iter = m_lut.begin();
  while (iter != m_lut.end()) {
    int sp = (*iter).first.first;
    if (sp != spold) {
      cout << endl;
      cout << "Qualified Patterns for Sector Processor " << setw(1) << sp << " :" << endl;
      cout << "===========================================" << endl;
      cout << endl;
      spold = sp;
    }
    cout << setw(2) << (*iter).first.second << "  " << setw(3) << (*iter).second.first << "  " << setw(5)
         << (*iter).second.second.size() << " : ";
    const vector<short>& patternlist = (*iter).second.second;
    vector<short>::const_iterator it;
    for (it = patternlist.begin(); it != patternlist.end(); it++) {
      cout << setw(5) << (*it) << " ";
    }
    cout << endl;
    iter++;
  }

  cout << endl;
}

//
// get coarse eta value for a given sector processor [1-6] and address [1-22]
//
int L1MuDTQualPatternLut::getCoarseEta(int sp, int adr) const {
  LUT::const_iterator it = m_lut.find(make_pair(sp, adr));
  if (it == m_lut.end()) {
    cerr << "Error: L1MuDTQualPatternLut: no coarse eta found for address " << adr << endl;
    return 0;
  }
  return (*it).second.first;
}

//
// get list of qualified patterns for a given sector processor [1-6] and address [1-22]
//
const vector<short>& L1MuDTQualPatternLut::getQualifiedPatterns(int sp, int adr) const {
  LUT::const_iterator it = m_lut.find(make_pair(sp, adr));
  if (it == m_lut.end()) {
    cerr << "Error: L1MuDTQualPatternLut: no pattern list found for address " << adr << endl;
  }
  return (*it).second.second;
}
