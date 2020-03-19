#ifndef CondFormats_EcalObjects_EcalWeightSet_HH
#define CondFormats_EcalObjects_EcalWeightSet_HH
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Container persistent object
 *  all weight objects needed to compute the pulse shape 
 *  with the weight method should go in this container
 *
 **/

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/Math/interface/Matrix.h"
#include <iostream>

class EcalWeightSet {
public:
  typedef math::Matrix<3, 10>::type EcalWeightMatrix;
  typedef math::Matrix<10, 10>::type EcalChi2WeightMatrix;

  EcalWeightSet();
  EcalWeightSet(const EcalWeightSet& aset);
  ~EcalWeightSet();

  EcalWeightMatrix& getWeightsBeforeGainSwitch() { return wgtBeforeSwitch_; }
  EcalWeightMatrix& getWeightsAfterGainSwitch() { return wgtAfterSwitch_; }
  EcalChi2WeightMatrix& getChi2WeightsBeforeGainSwitch() { return wgtChi2BeforeSwitch_; }
  EcalChi2WeightMatrix& getChi2WeightsAfterGainSwitch() { return wgtChi2AfterSwitch_; }

  const EcalWeightMatrix& getWeightsBeforeGainSwitch() const { return wgtBeforeSwitch_; }
  const EcalWeightMatrix& getWeightsAfterGainSwitch() const { return wgtAfterSwitch_; }
  const EcalChi2WeightMatrix& getChi2WeightsBeforeGainSwitch() const { return wgtChi2BeforeSwitch_; }
  const EcalChi2WeightMatrix& getChi2WeightsAfterGainSwitch() const { return wgtChi2AfterSwitch_; }

  EcalWeightSet& operator=(const EcalWeightSet& rhs);

  void print(std::ostream& o) const {
    using namespace std;
    o << "wgtBeforeSwitch_.: " << wgtBeforeSwitch_ << " wgtAfterSwitch_.: " << wgtAfterSwitch_
      << " wgtChi2BeforeSwitch_.: " << wgtChi2BeforeSwitch_ << " wgtChi2AfterSwitch_.: " << wgtChi2AfterSwitch_ << endl;
  }

private:
  EcalWeightMatrix wgtBeforeSwitch_;
  EcalWeightMatrix wgtAfterSwitch_;
  EcalChi2WeightMatrix wgtChi2BeforeSwitch_;
  EcalChi2WeightMatrix wgtChi2AfterSwitch_;

  COND_SERIALIZABLE;
};

#endif
