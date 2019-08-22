//-------------------------------------------------
//
//   Class L1MuTMChambPhContainer
//
//
//
//
//   Author List: Jorge Troconiz, George Karathanasis
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/L1TTwinMux/interface/L1MuTMChambPhContainer.h"

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
L1MuTMChambPhContainer::L1MuTMChambPhContainer() {}

//--------------
// Destructor --
//--------------
L1MuTMChambPhContainer::~L1MuTMChambPhContainer() {}

//--------------
// Operations --
//--------------
void L1MuTMChambPhContainer::setContainer(const Phi_Container& inputSegments) { phiSegments = inputSegments; }

L1MuTMChambPhContainer::Phi_Container const* L1MuTMChambPhContainer::getContainer() const { return &phiSegments; }

bool L1MuTMChambPhContainer::bxEmpty(int step) const {
  for (Phi_iterator i = phiSegments.begin(); i != phiSegments.end(); i++) {
    if (step == i->bxNum())
      return false;
  }

  return true;
}

int L1MuTMChambPhContainer::bxSize(int step1, int step2) const {
  int size = 0;

  for (Phi_iterator i = phiSegments.begin(); i != phiSegments.end(); i++) {
    if (step1 <= i->bxNum() && step2 >= i->bxNum() && i->Ts2Tag() == 0 && i->code() != 7)
      size++;
    if (step1 <= i->bxNum() - 1 && step2 >= i->bxNum() - 1 && i->Ts2Tag() == 1 && i->code() != 7)
      size++;
  }

  return (size);
}

L1MuDTChambPhDigi const* L1MuTMChambPhContainer::chPhiSegm1(int wheel, int stat, int sect, int step) const {
  L1MuDTChambPhDigi const* rT = nullptr;

  for (Phi_iterator i = phiSegments.begin(); i != phiSegments.end(); i++) {
    if (step == i->bxNum() && wheel == i->whNum() && sect == i->scNum() && stat == i->stNum() && i->Ts2Tag() == 0)
      rT = &(*i);
  }

  return (rT);
}

L1MuDTChambPhDigi const* L1MuTMChambPhContainer::chPhiSegm2(int wheel, int stat, int sect, int step) const {
  L1MuDTChambPhDigi const* rT = nullptr;

  for (Phi_iterator i = phiSegments.begin(); i != phiSegments.end(); i++) {
    if (step == i->bxNum() - 1 && wheel == i->whNum() && sect == i->scNum() && stat == i->stNum() && i->Ts2Tag() == 1)
      rT = &(*i);
  }

  return (rT);
}

L1MuDTChambPhDigi* L1MuTMChambPhContainer::chPhiSegm(int wheel, int stat, int sect, int step, int ts2tag) {
  L1MuDTChambPhDigi* rT = nullptr;
  for (Phi_Container::iterator i = phiSegments.begin(); i != phiSegments.end(); i++) {
    if (step == i->bxNum() && wheel == i->whNum() && sect == i->scNum() && stat == i->stNum() && i->Ts2Tag() == ts2tag)
      rT = &(*i);
  }

  return (rT);
}
