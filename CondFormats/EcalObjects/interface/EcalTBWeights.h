#ifndef CondFormats_EcalObjects_EcalTBWeights_H
#define CondFormats_EcalObjects_EcalTBWeights_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"

class EcalTBWeights {
public:
  typedef int EcalTDCId;
  typedef std::map<std::pair<EcalXtalGroupId, EcalTDCId>, EcalWeightSet> EcalTBWeightMap;

  EcalTBWeights();
  ~EcalTBWeights();

  // modifiers
  void setValue(const EcalXtalGroupId& groupId, const EcalTDCId& tdcId, const EcalWeightSet& weight);
  void setValue(const std::pair<EcalXtalGroupId, EcalTDCId>& keyPair, const EcalWeightSet& weight);

  // accessors
  const EcalTBWeightMap& getMap() const { return map_; }

private:
  EcalTBWeightMap map_;

  COND_SERIALIZABLE;
};
#endif
