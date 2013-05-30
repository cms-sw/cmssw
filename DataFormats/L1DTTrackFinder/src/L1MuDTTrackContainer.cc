//-------------------------------------------------
//
//   Class L1MuDTTrackContainer
//
//   Description: output data for DTTF trigger
//
//
//   Author List: Jorge Troconiz  UAM Madrid
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

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
L1MuDTTrackContainer::L1MuDTTrackContainer() {}

//--------------
// Destructor --
//--------------
L1MuDTTrackContainer::~L1MuDTTrackContainer() {}

//--------------
// Operations --
//--------------
void L1MuDTTrackContainer::setContainer(const TrackContainer& inputTracks) {

  dtTracks = inputTracks;
}

L1MuDTTrackContainer::TrackContainer* L1MuDTTrackContainer::getContainer() const {

  TrackContainer* rT=0;

  rT = const_cast<TrackContainer*>(&dtTracks);

  return(rT);
}

bool L1MuDTTrackContainer::bxEmpty(int step) const {

  bool empty = true;

  for ( Trackiterator i  = dtTracks.begin();
                      i != dtTracks.end();
                      i++ ) {
    if  (step == i->bx()) empty = false;
  }

  return(empty);
}

int L1MuDTTrackContainer::bxSize(int step1, int step2) const {

  int size = 0;

  for ( Trackiterator i  = dtTracks.begin();
                      i != dtTracks.end();
                      i++ ) {
    if  (step1 <= i->bx() && step2 >= i->bx() 
      && i->quality_packed() != 0) size++;
  }

  return(size);
}

L1MuDTTrackCand* L1MuDTTrackContainer::dtTrackCand1(int wheel, int sect, int step) const {

  L1MuDTTrackCand* rT=0;

  for ( Trackiterator i  = dtTracks.begin();
                      i != dtTracks.end();
                      i++ ) {
    if  (step == i->bx() && wheel == i->whNum() && sect == i->scNum()
      && i->TrkTag() == 0)
      rT = const_cast<L1MuDTTrackCand*>(&(*i));
  }

  return(rT);
}

L1MuDTTrackCand* L1MuDTTrackContainer::dtTrackCand2(int wheel, int sect, int step) const {

  L1MuDTTrackCand* rT=0;

  for ( Trackiterator i  = dtTracks.begin();
                      i != dtTracks.end();
                      i++ ) {
    if  (step == i->bx() && wheel == i->whNum() && sect == i->scNum()
      && i->TrkTag() == 1)
      rT = const_cast<L1MuDTTrackCand*>(&(*i));
  }

  return(rT);
}
