//-------------------------------------------------
//
//   Class L1Phase2MuDTExtPhContainer
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Nicolo' Trevisani - Oviedo ICTEA
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhContainer.h"

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
L1Phase2MuDTExtPhContainer::L1Phase2MuDTExtPhContainer() {}

//--------------
// Operations --
//--------------
void L1Phase2MuDTExtPhContainer::setContainer(const Segment_Container& inputSegments) { m_segments = inputSegments; }

L1Phase2MuDTExtPhContainer::Segment_Container const* L1Phase2MuDTExtPhContainer::getContainer() const {
  return &m_segments;
}
