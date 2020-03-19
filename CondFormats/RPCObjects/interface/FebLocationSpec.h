#ifndef CondFormatsRPCObjectsFebLocationSpec_H
#define CondFormatsRPCObjectsFebLocationSpec_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>

/** \class FebLocationSpec 
 * RPC FEB specification for readout decoding
 */

struct FebLocationSpec {
  char cmsEtaPartition;
  char positionInCmsEtaPartition;
  char localEtaPartition;
  char positionInLocalEtaPartition;

  /// debug printout
  std::string localEtaPartitionName() const;
  std::string print(int depth = 0) const;

  COND_SERIALIZABLE;
};
#endif
