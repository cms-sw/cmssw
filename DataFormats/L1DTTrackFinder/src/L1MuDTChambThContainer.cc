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
L1MuDTChambThContainer::L1MuDTChambThContainer() {}

//--------------
// Destructor --
//--------------
L1MuDTChambThContainer::~L1MuDTChambThContainer() {}

//--------------
// Operations --
//--------------
void L1MuDTChambThContainer::setContainer(const The_Container& inputSegments) {

  theSegments = inputSegments;
}

L1MuDTChambThContainer::The_Container* L1MuDTChambThContainer::getContainer() const {

  The_Container* rT=0;

  rT = const_cast<The_Container*>(&theSegments);

  return(rT);
}

bool L1MuDTChambThContainer::bxEmpty(int step) const {

  bool empty = true;

  for ( The_iterator i  = theSegments.begin();
                     i != theSegments.end();
                     i++ ) {
    if  (step == i->bxNum()) empty = false;
  }

  return(empty);
}

int L1MuDTChambThContainer::bxSize(int step1, int step2) const {

  int size = 0;

  for ( The_iterator i  = theSegments.begin();
                     i != theSegments.end();
                     i++ ) {
    if  (step1 <= i->bxNum() && step2 >= i->bxNum()) size++;
  }

  return(size);
}

L1MuDTChambThDigi* L1MuDTChambThContainer::chThetaSegm(int wheel, int stat, int sect, int step) const {

  L1MuDTChambThDigi* rT=0;

  for ( The_iterator i  = theSegments.begin();
                     i != theSegments.end();
                     i++ ) {
    if  (step == i->bxNum() && wheel == i->whNum() && sect == i->scNum()
      && stat == i->stNum() )
      rT = const_cast<L1MuDTChambThDigi*>(&(*i));
  }

  return(rT);
}

