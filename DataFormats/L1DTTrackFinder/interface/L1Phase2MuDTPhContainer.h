//-------------------------------------------------
//
//   Class L1Phase2MuDTPhContainer
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Federica Primavera  Bologna INFN
//
//
//--------------------------------------------------
#ifndef L1Phase2MuDTPhContainer_H
#define L1Phase2MuDTPhContainer_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"

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

class L1Phase2MuDTPhContainer {
public:
  typedef std::vector<L1Phase2MuDTPhDigi> Segment_Container;
  typedef Segment_Container::const_iterator Segment_iterator;

  //  Constructor
  L1Phase2MuDTPhContainer();

  void setContainer(const Segment_Container& inputSegments);

  Segment_Container const* getContainer() const;

private:
  Segment_Container m_segments;
};

#endif
