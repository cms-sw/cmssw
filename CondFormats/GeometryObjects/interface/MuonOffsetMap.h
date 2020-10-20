#ifndef CondFormats_GeometryObjects_MuonOffsetMap_h
#define CondFormats_GeometryObjects_MuonOffsetMap_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <string>
#include <vector>
#include <unordered_map>

class MuonOffsetMap {
public:
  MuonOffsetMap() = default;
  ~MuonOffsetMap() = default;

  std::unordered_map<std::string, std::pair<int, int> > muonMap_;

  COND_SERIALIZABLE;
};

#endif
