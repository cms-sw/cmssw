//-------------------------------------------------
//
//   Class: L1MuDTSecProcMap
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

#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcMap.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/DTTrackFinder/interface/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSectorProcessor.h"

using namespace std;

// --------------------------------
//       class L1MuDTSecProcMap
//---------------------------------

//----------------
// Constructors --
//----------------

L1MuDTSecProcMap::L1MuDTSecProcMap() : m_map() {}

//--------------
// Destructor --
//--------------

L1MuDTSecProcMap::~L1MuDTSecProcMap() = default;

//--------------
// Operations --
//--------------

//
// return Sector Processor
//
const L1MuDTSectorProcessor* L1MuDTSecProcMap::sp(const L1MuDTSecProcId& id) const {
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
void L1MuDTSecProcMap::insert(const L1MuDTSecProcId& id, std::unique_ptr<L1MuDTSectorProcessor> sp) {
  //SPmap::const_iterator it = m_map.find(id);
  //  if ( it != m_map.end() )
  //    cerr << "Error: More than one Sector Processor with same identifier"
  //         << endl;
  m_map[id] = std::move(sp);
}
