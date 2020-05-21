//-------------------------------------------------
//
//   Class L1MuDTChambPhContainer
//
//   Description: input data for PHTF trigger
//
//
//   Author List: Jorge Troconiz  UAM Madrid
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"

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
L1MuDTChambPhContainer::L1MuDTChambPhContainer(Phi_Container c) : phiSegments(std::move(c)) {}

//--------------
// Operations --
//--------------
void L1MuDTChambPhContainer::setContainer(Phi_Container inputSegments) { phiSegments = std::move(inputSegments); }

L1MuDTChambPhContainer::Phi_Container const* L1MuDTChambPhContainer::getContainer() const { return &phiSegments; }

bool L1MuDTChambPhContainer::bxEmpty(int step) const {
  bool empty = true;

  for (const auto& phiSegment : phiSegments) {
    if (step == phiSegment.bxNum())
      empty = false;
  }

  return (empty);
}

int L1MuDTChambPhContainer::bxSize(int step1, int step2) const {
  int size = 0;

  for (const auto& phiSegment : phiSegments) {
    if (step1 <= phiSegment.bxNum() && step2 >= phiSegment.bxNum() && phiSegment.Ts2Tag() == 0 &&
        phiSegment.code() != 7)
      size++;
    if (step1 <= phiSegment.bxNum() - 1 && step2 >= phiSegment.bxNum() - 1 && phiSegment.Ts2Tag() == 1 &&
        phiSegment.code() != 7)
      size++;
  }

  return (size);
}

L1MuDTChambPhDigi const* L1MuDTChambPhContainer::chPhiSegm1(int wheel, int stat, int sect, int step) const {
  L1MuDTChambPhDigi const* rT = nullptr;

  for (const auto& phiSegment : phiSegments) {
    if (step == phiSegment.bxNum() && wheel == phiSegment.whNum() && sect == phiSegment.scNum() &&
        stat == phiSegment.stNum() && phiSegment.Ts2Tag() == 0)
      rT = &phiSegment;
  }

  return (rT);
}

L1MuDTChambPhDigi const* L1MuDTChambPhContainer::chPhiSegm2(int wheel, int stat, int sect, int step) const {
  L1MuDTChambPhDigi const* rT = nullptr;

  for (const auto& phiSegment : phiSegments) {
    if (step == phiSegment.bxNum() - 1 && wheel == phiSegment.whNum() && sect == phiSegment.scNum() &&
        stat == phiSegment.stNum() && phiSegment.Ts2Tag() == 1)
      rT = &phiSegment;
  }

  return (rT);
}
