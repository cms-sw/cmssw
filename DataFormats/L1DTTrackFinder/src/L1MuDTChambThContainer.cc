//-------------------------------------------------
//
//   Class L1MuDTChambThContainer
//
//   Description: input data for ETTF trigger
//
//
//   Author List: Jorge Troconiz  UAM Madrid
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

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
L1MuDTChambThContainer::L1MuDTChambThContainer(The_Container c) : theSegments{std::move(c)} {}

//--------------
// Operations --
//--------------
void L1MuDTChambThContainer::setContainer(The_Container inputSegments) { theSegments = std::move(inputSegments); }

L1MuDTChambThContainer::The_Container const* L1MuDTChambThContainer::getContainer() const { return &theSegments; }

bool L1MuDTChambThContainer::bxEmpty(int step) const {
  bool empty = true;

  for (const auto& theSegment : theSegments) {
    if (step == theSegment.bxNum())
      empty = false;
  }

  return (empty);
}

int L1MuDTChambThContainer::bxSize(int step1, int step2) const {
  int size = 0;

  for (const auto& theSegment : theSegments) {
    if (step1 <= theSegment.bxNum() && step2 >= theSegment.bxNum())
      size++;
  }

  return (size);
}

L1MuDTChambThDigi const* L1MuDTChambThContainer::chThetaSegm(int wheel, int stat, int sect, int step) const {
  L1MuDTChambThDigi const* rT = nullptr;

  for (const auto& theSegment : theSegments) {
    if (step == theSegment.bxNum() && wheel == theSegment.whNum() && sect == theSegment.scNum() &&
        stat == theSegment.stNum())
      rT = &theSegment;
  }

  return (rT);
}
