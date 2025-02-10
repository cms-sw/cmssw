//-------------------------------------------------
//
//   Class: L1MuBMSecProcMap
//
//   Description: Sector Processor container
//
//
//
//   Author :
//   N. Neumeister             CERN EP
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMSecProcMap.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "DataFormats/L1TMuon/interface/BMTF/L1MuBMSecProcId.h"
#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSectorProcessor.h"

using namespace std;

// --------------------------------
//       class L1MuBMSecProcMap
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuBMSecProcMap::L1MuBMSecProcMap() : m_map() {}

//--------------
// Operations --
//--------------

//
// return Sector Processor
//
L1MuBMSectorProcessor* L1MuBMSecProcMap::sp(const L1MuBMSecProcId& id) const {
  SPmap::const_iterator it = m_map.find(id);
  if (it == m_map.end()) {
    //    cerr << "Error: Sector Processor not in the map" << endl;
    return nullptr;
  }
  return (*it).second.get();
}

//
// insert Sector Processor into container
//
void L1MuBMSecProcMap::insert(const L1MuBMSecProcId& id, std::unique_ptr<L1MuBMSectorProcessor> sp) {
  //SPmap::const_iterator it = m_map.find(id);
  //  if ( it != m_map.end() )
  //    cerr << "Error: More than one Sector Processor with same identifier"
  //         << endl;
  m_map[id] = std::move(sp);
}
