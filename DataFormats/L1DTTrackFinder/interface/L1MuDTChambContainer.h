//-------------------------------------------------
//
//   Class L1MuDTChambContainer
//
//   Description: input data for Phase2 trigger
//
//
//   Author List: Federica Primavera  Bologna INFN
//
//
//--------------------------------------------------
#ifndef L1MuDTChambContainer_H
#define L1MuDTChambContainer_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "../interface/L1MuDTChambDigi.h"

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


class L1MuDTChambContainer {

 public:

  typedef std::vector<L1MuDTChambDigi>  Segment_Container;
  typedef Segment_Container::const_iterator   Segment_iterator;

  //  Constructors
  L1MuDTChambContainer();

  //  Destructor
  ~L1MuDTChambContainer();

  void setContainer(const Segment_Container& inputSegments);

  Segment_Container const* getContainer() const;


 private:

  Segment_Container m_segments;

};

#endif
