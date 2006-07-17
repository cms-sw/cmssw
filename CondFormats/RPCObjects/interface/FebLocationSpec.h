#ifndef CondFormatsRPCObjectsFebLocationSpec_H
#define CondFormatsRPCObjectsFebLocationSpec_H

#include <string>

/** \class FebLocationSpec 
 * RPC FEB specification for readout decoding
 */

struct FebLocationSpec {

  std::string cmsEtaPartition;
  int positionInCmsEtaPartition;
  std::string localEtaPartition;
  int positionInLocalEtaPartition;

  /// debug printout
  void print(int depth = 0) const;
};
#endif
