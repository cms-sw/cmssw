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

  for (The_iterator i = theSegments.begin(); i != theSegments.end(); i++) {
    if (step == i->bxNum())
      empty = false;
  }

  return (empty);
}

int L1MuDTChambThContainer::bxSize(int step1, int step2) const {
  int size = 0;

  for (The_iterator i = theSegments.begin(); i != theSegments.end(); i++) {
    if (step1 <= i->bxNum() && step2 >= i->bxNum())
      size++;
  }

  return (size);
}

L1MuDTChambThDigi const* L1MuDTChambThContainer::chThetaSegm(int wheel, int stat, int sect, int step) const {
  L1MuDTChambThDigi const* rT = nullptr;

  for (The_iterator i = theSegments.begin(); i != theSegments.end(); i++) {
    if (step == i->bxNum() && wheel == i->whNum() && sect == i->scNum() && stat == i->stNum())
      rT = &(*i);
  }

  return (rT);
}
