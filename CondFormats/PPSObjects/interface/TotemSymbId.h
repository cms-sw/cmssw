/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#ifndef CondFormats_PPSObjects_TotemSymbId
#define CondFormats_PPSObjects_TotemSymbId

#include "CondFormats/Serialization/interface/Serializable.h"
#include <iostream>

/**
 *\brief Symbolic ID describing an entity of a TOTEM subdetector.
 **/
class TotemSymbID {
public:
  /// chip ID, raw integer representation of DetId class
  unsigned int symbolicID;

  void print(std::ostream &os, std::string subSystemName) const;

  bool operator<(const TotemSymbID &sid) const { return (symbolicID < sid.symbolicID); }

  bool operator==(const TotemSymbID &sid) const { return (symbolicID == sid.symbolicID); }

  friend std::ostream &operator<<(std::ostream &s, const TotemSymbID &sid);

  COND_SERIALIZABLE;
};

#endif
