//-------------------------------------------------
//
//   Class L1Phase2MuDTExtPhiThetaPairContainer
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: J. Fernandez - Oviedo ICTEA
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhiThetaPairContainer.h"

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
L1Phase2MuDTExtPhiThetaPairContainer::L1Phase2MuDTExtPhiThetaPairContainer() {}

//--------------
// Operations --
//--------------
void L1Phase2MuDTExtPhiThetaPairContainer::setContainer(const Segment_Container& inputSegments) {
  m_segments = inputSegments;
}

L1Phase2MuDTExtPhiThetaPairContainer::Segment_Container const* L1Phase2MuDTExtPhiThetaPairContainer::getContainer()
    const {
  return &m_segments;
}
