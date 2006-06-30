#ifndef CondFormatsRPCObjectsDBSpecToDetUnit_H
#define CondFormatsRPCObjectsDBSpecToDetUnit_H

#include <string>
#include "DataFormats/DetId/interface/DetId.h"
class ChamberLocationSpec;

class DBSpecToDetUnit {
public:
  uint32_t operator()(const ChamberLocationSpec & location, const std::string & roll);
};
#endif
