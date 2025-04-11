//-------------------------------------------------
//
//   Class L1Phase2MuDTShowerContainer
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger shower
//
//
//   Author List: Daniel Estrada Acevedo Oviedo Spain
//
//
//--------------------------------------------------
#ifndef L1Phase2MuDTShowerContainer_H
#define L1Phase2MuDTShowerContainer_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTShower.h"

//----------------------
// Base Class Headers --
//----------------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1Phase2MuDTShowerContainer {
public:
  typedef std::vector<L1Phase2MuDTShower> Shower_Container;
  typedef Shower_Container::const_iterator Shower_iterator;

  //  Constructor
  L1Phase2MuDTShowerContainer();

  void setContainer(const Shower_Container& inputShowers);

  Shower_Container const* getContainer() const;

private:
  Shower_Container m_showers;
};

#endif
