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
#ifndef L1MuDTChambThContainer_H
#define L1MuDTChambThContainer_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"

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

class L1MuDTChambThContainer {
public:
  typedef std::vector<L1MuDTChambThDigi> The_Container;
  typedef The_Container::const_iterator The_iterator;

  //  Constructors
  L1MuDTChambThContainer() = default;
  L1MuDTChambThContainer(The_Container);

  //  Destructor
  ~L1MuDTChambThContainer() = default;

  void setContainer(The_Container inputSegments);

  The_Container const* getContainer() const;

  bool bxEmpty(int step) const;

  int bxSize(int step1, int step2) const;

  L1MuDTChambThDigi const* chThetaSegm(int wheel, int stat, int sect, int bx) const;

private:
  The_Container theSegments;
};

#endif
