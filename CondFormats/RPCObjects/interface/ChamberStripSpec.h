#ifndef CondFormatsRPCObjectsChamberStripSpec_H
#define CondFormatsRPCObjectsChamberStripSpec_H

/** \class ChamberStripSpec
 * RPC strip specification for readout decoding
 */
#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>

struct ChamberStripSpec {
  int cablePinNumber;
  int chamberStripNumber;
  int cmsStripNumber;
  
  /// debug printout
  std::string print( int depth = 0) const;

  COND_SERIALIZABLE;
};

#endif
