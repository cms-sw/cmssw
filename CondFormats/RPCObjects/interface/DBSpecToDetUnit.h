#ifndef CondFormatsRPCObjectsDBSpecToDetUnit_H
#define CondFormatsRPCObjectsDBSpecToDetUnit_H

#include "DataFormats/DetId/interface/DetId.h"
struct ChamberLocationSpec;
struct FebLocationSpec;

class DBSpecToDetUnit {
public:
  uint32_t operator()(const ChamberLocationSpec & location, const FebLocationSpec & feb);
};
#endif
