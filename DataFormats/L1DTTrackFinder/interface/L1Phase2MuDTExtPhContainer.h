//-------------------------------------------------
//
//   Class L1Phase2MuDTExtPhContainer
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Nicolo' Trevisani - Oviedo ICTEA
//
//
//--------------------------------------------------
#ifndef L1Phase2MuDTExtPhContainer_H
#define L1Phase2MuDTExtPhContainer_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhDigi.h"

//----------------------
// Base Class Headers --
//----------------------
#include <vector>

//---------------
// C++ Headers --
//---------------

//---------------------
//-- Class Interface --
//---------------------

class L1Phase2MuDTExtPhContainer {
public:
  typedef std::vector<L1Phase2MuDTExtPhDigi> Segment_Container;
  typedef Segment_Container::const_iterator Segment_iterator;

  //  Constructor
  L1Phase2MuDTExtPhContainer();

  void setContainer(const Segment_Container& inputSegments);

  Segment_Container const* getContainer() const;

private:
  Segment_Container m_segments;
};

#endif
