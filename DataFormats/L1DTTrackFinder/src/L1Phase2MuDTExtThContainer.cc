//-------------------------------------------------
//
//   Class L1Phase2MuDTExtThContainer
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
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThContainer.h"

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
L1Phase2MuDTExtThContainer::L1Phase2MuDTExtThContainer() {}

//--------------
// Operations --
//--------------
void L1Phase2MuDTExtThContainer::setContainer(const Segment_Container& inputSegments) { m_segments = inputSegments; }

L1Phase2MuDTExtThContainer::Segment_Container const* L1Phase2MuDTExtThContainer::getContainer() const {
  return &m_segments;
}
