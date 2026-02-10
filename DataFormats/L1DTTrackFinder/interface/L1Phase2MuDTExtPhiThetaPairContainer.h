//-------------------------------------------------
//
//   Class L1Phase2MuDTExtPhiThetaPairContainer
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: J. Fernandez - Oviedo ICTEA
//
//
//--------------------------------------------------
#ifndef L1Phase2MuDTExtPhiThetaPairContainer_H
#define L1Phase2MuDTExtPhiThetaPairContainer_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhiThetaPair.h"

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

class L1Phase2MuDTExtPhiThetaPairContainer {
public:
  typedef std::vector<L1Phase2MuDTExtPhiThetaPair> Segment_Container;
  typedef Segment_Container::const_iterator Segment_iterator;

  //  Constructor
  L1Phase2MuDTExtPhiThetaPairContainer();

  void setContainer(const Segment_Container& inputSegments);

  Segment_Container const& getContainer() const;

private:
  Segment_Container m_segments;
};

#endif
