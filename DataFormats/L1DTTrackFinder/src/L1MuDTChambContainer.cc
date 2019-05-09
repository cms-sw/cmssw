//-------------------------------------------------
//
//   Class L1MuDTChambContainer
//
//   Description: input data for Phase2 trigger
//
//
//   Author List: Federica Primavera  Bologna INFN
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "../interface/L1MuDTChambContainer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------
using namespace std;

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
L1MuDTChambContainer::L1MuDTChambContainer() {}

//--------------
// Destructor --
//--------------
L1MuDTChambContainer::~L1MuDTChambContainer() {}

//--------------
// Operations --
//--------------
void L1MuDTChambContainer::setContainer(const Segment_Container& inputSegments) {

  m_segments = inputSegments;
}

L1MuDTChambContainer::Segment_Container const* L1MuDTChambContainer::getContainer() const {
  return &m_segments;
}
