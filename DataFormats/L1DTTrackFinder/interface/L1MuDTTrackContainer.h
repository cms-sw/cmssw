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
#ifndef L1MuDTTrackContainer_H
#define L1MuDTTrackContainer_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackCand.h"

//----------------------
// Base Class Headers --
//----------------------
#include <vector>

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------


class L1MuDTTrackContainer {

 public:

  typedef std::vector<L1MuDTTrackCand>    TrackContainer;
  typedef TrackContainer::const_iterator  Trackiterator;
  typedef TrackContainer::iterator        TrackIterator;

  //  Constructors
  L1MuDTTrackContainer();

  //  Destructor
  ~L1MuDTTrackContainer();

  void setContainer(const TrackContainer& inputTracks);

  TrackContainer* getContainer() const;

  bool bxEmpty(int step) const;

  int bxSize(int step1, int step2) const;

  L1MuDTTrackCand* dtTrackCand1(int wheel, int sect, int bx) const;

  L1MuDTTrackCand* dtTrackCand2(int wheel, int sect, int bx) const;


 private:

  TrackContainer dtTracks; 

};

#endif
