#ifndef CondFormatsRPCObjectsChamberLocationSpec_H
#define CondFormatsRPCObjectsChamberLocationSpec_H
#include "CondFormats/Serialization/interface/Serializable.h"

#include<string>

/* \class ChamberLocationSpec
 * Chamber Location specification as in online DB
 */

struct ChamberLocationSpec {
  int diskOrWheel;
  int layer;
  int sector;
  char subsector;
  char febZOrnt;
  char febZRadOrnt;
  char barrelOrEndcap; 

  /// debug printout
  std::string print( int depth = 0) const;

  std::string chamberLocationName() const;


  COND_SERIALIZABLE;
};


#endif
