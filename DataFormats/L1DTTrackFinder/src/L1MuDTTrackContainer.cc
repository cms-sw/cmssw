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

L1MuDTTrackContainer::TrackContainer* L1MuDTTrackContainer::getContainer() const {

  TrackContainer* rT=0;

  rT = const_cast<TrackContainer*>(&dtTracks);

  return(rT);
}

//--------------
// Operations --
//--------------
void L1MuDTTrackContainer::setContainer(TrackContainer inputTracks) {

  dtTracks = inputTracks;
}
