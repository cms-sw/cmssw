#ifndef CondFormats_EcalObjects_EcalWeightSet_HH
#define CondFormats_EcalObjects_EcalWeightSet_HH
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Container persistent object
 *  all weight objects needed to compute the pulse shape 
 *  with the weight method should go in this container
 *
 **/

#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/Math/interface/EcalWeight.h"
#include <iostream>

class EcalWeightSet {

  public:
  
    EcalWeightSet();
    EcalWeightSet(const EcalWeightSet& aset);
    ~EcalWeightSet();

    math::EcalWeightMatrix::type& getWeightsBeforeGainSwitch() { return wgtBeforeSwitch_; }
    math::EcalWeightMatrix::type& getWeightsAfterGainSwitch()  { return wgtAfterSwitch_; }
    math::EcalChi2WeightMatrix::type& getChi2WeightsBeforeGainSwitch()             { return wgtChi2BeforeSwitch_; }
    math::EcalChi2WeightMatrix::type& getChi2WeightsAfterGainSwitch()             { return wgtChi2AfterSwitch_; }
    
    const math::EcalWeightMatrix::type& getWeightsBeforeGainSwitch() const { return wgtBeforeSwitch_; }
    const math::EcalWeightMatrix::type& getWeightsAfterGainSwitch()  const { return wgtAfterSwitch_; }
    const math::EcalChi2WeightMatrix::type& getChi2WeightsBeforeGainSwitch() const { return wgtChi2BeforeSwitch_; }
    const math::EcalChi2WeightMatrix::type& getChi2WeightsAfterGainSwitch() const { return wgtChi2AfterSwitch_; }
    
    EcalWeightSet& operator=(const EcalWeightSet& rhs);
    
    void print(std::ostream& o) const {
      using namespace std;
      o << "wgtBeforeSwitch_.: " << wgtBeforeSwitch_
	<< " wgtAfterSwitch_.: " << wgtAfterSwitch_
	<< " wgtChi2BeforeSwitch_.: " << wgtChi2BeforeSwitch_
	<< " wgtChi2AfterSwitch_.: " << wgtChi2AfterSwitch_
	<< endl;
    }
    
    
 private:
    math::EcalWeightMatrix::type wgtBeforeSwitch_;
    math::EcalWeightMatrix::type wgtAfterSwitch_;
    math::EcalChi2WeightMatrix::type wgtChi2BeforeSwitch_;
    math::EcalChi2WeightMatrix::type wgtChi2AfterSwitch_;
};

#endif
