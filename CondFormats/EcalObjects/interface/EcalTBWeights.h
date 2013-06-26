#ifndef CondFormats_EcalObjects_EcalTBWeights_H
#define CondFormats_EcalObjects_EcalTBWeights_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalTBWeights.h,v 1.2 2006/02/23 16:56:34 rahatlou Exp $
 **/


#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"


class EcalTBWeights {
  public:
   typedef int EcalTDCId;
   typedef std::map< std::pair< EcalXtalGroupId, EcalTDCId >, EcalWeightSet > EcalTBWeightMap;

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
