//-------------------------------------------------
//
//   Class L1MuTMChambPhContainer
//
//   Description: input data for PHTF trigger
//
//
//   Author List: Jorge Troconiz, George Karathanasis
//
//
//--------------------------------------------------
#ifndef L1T_TwinMux_L1MuTMChambPhContainer_H
#define L1T_TwinMux_L1MuTMChambPhContainer_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"

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


class L1MuTMChambPhContainer {

 public:

  typedef std::vector<L1MuDTChambPhDigi>  Phi_Container;
  typedef Phi_Container::const_iterator   Phi_iterator;

  //  Constructors
  L1MuTMChambPhContainer();

  //  Destructor
  ~L1MuTMChambPhContainer();

  void setContainer(const Phi_Container& inputSegments);

  Phi_Container const* getContainer() const;

  bool bxEmpty(int step) const;

  int bxSize(int step1, int step2) const;

  L1MuDTChambPhDigi const* chPhiSegm1(int wheel, int stat, int sect, int bx) const;

  L1MuDTChambPhDigi const* chPhiSegm2(int wheel, int stat, int sect, int bx) const;

  L1MuDTChambPhDigi* chPhiSegm(int wheel, int stat, int sect, int bx, int ts2tag);

 private:

  Phi_Container phiSegments; 

};

#endif
