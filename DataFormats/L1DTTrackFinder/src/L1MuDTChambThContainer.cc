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
void L1MuDTChambThContainer::setContainer(The_Container inputSegments) {

  theSegments = inputSegments;
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

