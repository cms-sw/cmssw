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
L1MuDTChambPhContainer::L1MuDTChambPhContainer() {}

//--------------
// Destructor --
//--------------
L1MuDTChambPhContainer::~L1MuDTChambPhContainer() {}

//--------------
// Operations --
//--------------
void L1MuDTChambPhContainer::setContainer(const Phi_Container& inputSegments) {

  phiSegments = inputSegments;
}

L1MuDTChambPhContainer::Phi_Container* L1MuDTChambPhContainer::getContainer() const {

  Phi_Container* rT=0;

  rT = const_cast<Phi_Container*>(&phiSegments);

  return(rT);
}

bool L1MuDTChambPhContainer::bxEmpty(int step) const {

  bool empty = true;

  for ( Phi_iterator i  = phiSegments.begin();
                     i != phiSegments.end();
                     i++ ) {
    if  (step == i->bxNum()) empty = false;
  }

  return(empty);
}

int L1MuDTChambPhContainer::bxSize(int step1, int step2) const {

  int size = 0;

  for ( Phi_iterator i  = phiSegments.begin();
                     i != phiSegments.end();
                     i++ ) {
    if  (step1 <= i->bxNum() && step2 >= i->bxNum() 
      && i->Ts2Tag() == 0 && i->code() != 7) size++;
    if  (step1 <= i->bxNum()-1 && step2 >= i->bxNum()-1 
      && i->Ts2Tag() == 1 && i->code() != 7) size++;
  }

  return(size);
}

L1MuDTChambPhDigi* L1MuDTChambPhContainer::chPhiSegm1(int wheel, int stat, int sect, int step) const {

  L1MuDTChambPhDigi* rT=0;

  for ( Phi_iterator i  = phiSegments.begin();
                     i != phiSegments.end();
                     i++ ) {
    if  (step == i->bxNum() && wheel == i->whNum() && sect == i->scNum()
      && stat == i->stNum() && i->Ts2Tag() == 0)
      rT = const_cast<L1MuDTChambPhDigi*>(&(*i));
  }

  return(rT);
}

L1MuDTChambPhDigi* L1MuDTChambPhContainer::chPhiSegm2(int wheel, int stat, int sect, int step) const {

  L1MuDTChambPhDigi* rT=0;

  for ( Phi_iterator i  = phiSegments.begin();
                     i != phiSegments.end();
                     i++ ) {
    if  (step == i->bxNum()-1 && wheel == i->whNum() && sect == i->scNum()
      && stat == i->stNum() && i->Ts2Tag() == 1)
      rT = const_cast<L1MuDTChambPhDigi*>(&(*i));
  }

  return(rT);
}
