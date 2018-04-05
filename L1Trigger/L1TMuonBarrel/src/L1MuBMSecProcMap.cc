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

#include "L1Trigger/L1TMuonBarrel/src/L1MuBMSecProcMap.h"

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
// Destructor --
//--------------

L1MuBMSecProcMap::~L1MuBMSecProcMap() {

  SPmap_iter iter = m_map.begin();
  while ( iter != m_map.end() ) {
    delete (*iter).second;
    iter++;
  }
  m_map.clear();

}


//--------------
// Operations --
//--------------

//
// return Sector Processor
//
L1MuBMSectorProcessor* L1MuBMSecProcMap::sp(const L1MuBMSecProcId& id ) const {

  SPmap::const_iterator it = m_map.find(id);
  if ( it == m_map.end() ) {
    //    cerr << "Error: Sector Processor not in the map" << endl;
    return nullptr;
  }
  return (*it).second;

}


//
// insert Sector Processor into container
//
void L1MuBMSecProcMap::insert(const L1MuBMSecProcId& id, L1MuBMSectorProcessor* sp)  {

  //SPmap::const_iterator it = m_map.find(id);
  //  if ( it != m_map.end() )
    //    cerr << "Error: More than one Sector Processor with same identifier"
    //         << endl;
  m_map[id] = sp;

}
