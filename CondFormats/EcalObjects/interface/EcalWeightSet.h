#ifndef CondFormats_EcalObjects_EcalWeightSet_HH
#define CondFormats_EcalObjects_EcalWeightSet_HH
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Container persistent object
 *  all weight objects needed to compute the pulse shape 
 *  with the weight method should go in this container
 *
 **/


#include <vector>
#include "CondFormats/EcalObjects/interface/EcalWeight.h"

typedef std::vector< std::vector< EcalWeight > > EcalWeightMatrix;

class EcalWeightSet {

  public:
    EcalWeightSet();
    ~EcalWeightSet();

    EcalWeightMatrix& getWeightsBeforeGainSwitch() { return wgtBeforeSwitch_; }
    EcalWeightMatrix& getWeightsAfterGainSwitch()  { return wgtAfterSwitch_; }
    EcalWeightMatrix& getChi2WeightsBeforeGainSwitch()             { return wgtChi2BeforeSwitch_; }
    EcalWeightMatrix& getChi2WeightsAfterGainSwitch()             { return wgtChi2AfterSwitch_; }

    const EcalWeightMatrix& getWeightsBeforeGainSwitch() const { return wgtBeforeSwitch_; }
    const EcalWeightMatrix& getWeightsAfterGainSwitch()  const { return wgtAfterSwitch_; }
    const EcalWeightMatrix& getChi2WeightsBeforeGainSwitch()             const { return wgtChi2BeforeSwitch_; }
    const EcalWeightMatrix& getChi2WeightsAfterGainSwitch()             const { return wgtChi2AfterSwitch_; }

  private:
     std::vector< std::vector< EcalWeight > > wgtBeforeSwitch_;
     std::vector< std::vector< EcalWeight > > wgtAfterSwitch_;
     std::vector< std::vector< EcalWeight > > wgtChi2BeforeSwitch_;
     std::vector< std::vector< EcalWeight > > wgtChi2AfterSwitch_;
};

#endif
