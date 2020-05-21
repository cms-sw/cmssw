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
void L1MuDTTrackContainer::setContainer(const TrackContainer& inputTracks) { dtTracks = inputTracks; }

L1MuDTTrackContainer::TrackContainer const* L1MuDTTrackContainer::getContainer() const { return &dtTracks; }

bool L1MuDTTrackContainer::bxEmpty(int step) const {
  bool empty = true;

  for (const auto& dtTrack : dtTracks) {
    if (step == dtTrack.bx())
      empty = false;
  }

  return (empty);
}

int L1MuDTTrackContainer::bxSize(int step1, int step2) const {
  int size = 0;

  for (const auto& dtTrack : dtTracks) {
    if (step1 <= dtTrack.bx() && step2 >= dtTrack.bx() && dtTrack.quality_packed() != 0)
      size++;
  }

  return (size);
}

L1MuDTTrackCand const* L1MuDTTrackContainer::dtTrackCand1(int wheel, int sect, int step) const {
  L1MuDTTrackCand const* rT = nullptr;

  for (const auto& dtTrack : dtTracks) {
    if (step == dtTrack.bx() && wheel == dtTrack.whNum() && sect == dtTrack.scNum() && dtTrack.TrkTag() == 0)
      rT = &dtTrack;
  }

  return (rT);
}

L1MuDTTrackCand const* L1MuDTTrackContainer::dtTrackCand2(int wheel, int sect, int step) const {
  L1MuDTTrackCand const* rT = nullptr;

  for (const auto& dtTrack : dtTracks) {
    if (step == dtTrack.bx() && wheel == dtTrack.whNum() && sect == dtTrack.scNum() && dtTrack.TrkTag() == 1)
      rT = &dtTrack;
  }

  return (rT);
}
