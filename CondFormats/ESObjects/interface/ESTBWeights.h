#ifndef CondFormats_ESObjects_ESTBWeights_H
#define CondFormats_ESObjects_ESTBWeights_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include "CondFormats/ESObjects/interface/ESStripGroupId.h"
#include "CondFormats/ESObjects/interface/ESWeightSet.h"

class ESTBWeights {
public:
  typedef std::map<ESStripGroupId, ESWeightSet> ESTBWeightMap;

  ESTBWeights();
  ~ESTBWeights();

  // modifiers
  void setValue(const ESStripGroupId& groupId, const ESWeightSet& weight);

  // accessors
  const ESTBWeightMap& getMap() const { return map_; }

private:
  ESTBWeightMap map_;

  COND_SERIALIZABLE;
};
#endif
