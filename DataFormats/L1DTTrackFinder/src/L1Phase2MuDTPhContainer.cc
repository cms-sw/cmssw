//-------------------------------------------------
//
//   Class L1MuDTChambContainer
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Federica Primavera  Bologna INFN
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
L1Phase2MuDTPhContainer::L1Phase2MuDTPhContainer() {}

//--------------
// Operations --
//--------------
void L1Phase2MuDTPhContainer::setContainer(const Segment_Container& inputSegments) { m_segments = inputSegments; }

L1Phase2MuDTPhContainer::Segment_Container const* L1Phase2MuDTPhContainer::getContainer() const { return &m_segments; }
