#ifndef CondFormats_EcalObjects_EcalTBWeights_H
#define CondFormats_EcalObjects_EcalTBWeights_H

#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"


typedef int EcalTDCId;
typedef std::map< std::pair< EcalXtalGroupId, EcalTDCId >, EcalWeightSet > EcalTBWeightMap;

class EcalTBWeights {
  public:
    EcalTBWeights();
    ~EcalTBWeights();

    // modifiers
    void setValue(const EcalXtalGroupId& groupId, const EcalTDCId& tdcId, const EcalWeightSet& weight);
    void setValue( const std::pair<EcalXtalGroupId,EcalTDCId >& keyPair, const EcalWeightSet& weight);

    // accessors
    const EcalTBWeightMap& getMap() const { return map_; }

  private:
    EcalTBWeightMap map_;
};
#endif
