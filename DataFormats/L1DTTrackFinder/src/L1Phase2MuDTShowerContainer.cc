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
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTShowerContainer.h"

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
L1Phase2MuDTShowerContainer::L1Phase2MuDTShowerContainer() {}

//--------------
// Operations --
//--------------
void L1Phase2MuDTShowerContainer::setContainer(const Shower_Container& inputShowers) { m_showers = inputShowers; }

L1Phase2MuDTShowerContainer::Shower_Container const* L1Phase2MuDTShowerContainer::getContainer() const { return &m_showers; }
