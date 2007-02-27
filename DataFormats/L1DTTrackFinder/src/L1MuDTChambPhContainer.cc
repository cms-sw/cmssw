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
void L1MuDTChambPhContainer::setContainer(Phi_Container inputSegments) {

  phiSegments = inputSegments;
}

L1MuDTChambPhContainer::Phi_Container* L1MuDTChambPhContainer::getContainer() const {

  Phi_Container* rT=0;

  rT = const_cast<Phi_Container*>(&phiSegments);

  return(rT);
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
