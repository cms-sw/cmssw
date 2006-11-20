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
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

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

  typedef std::vector<L1MuRegionalCand>   TrackContainer;
  typedef TrackContainer::const_iterator  Trackiterator;

  //  Constructors
  L1MuDTTrackContainer();

  //  Destructor
  ~L1MuDTTrackContainer();

  void setContainer(TrackContainer inputTracks);

  TrackContainer* getContainer() const;

 private:

  TrackContainer dtTracks; 

};

#endif
