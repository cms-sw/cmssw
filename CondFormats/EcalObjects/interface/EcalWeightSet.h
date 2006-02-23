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
#include <iostream>

class EcalWeightSet {

  public:
    typedef std::vector< std::vector< EcalWeight > > EcalWeightMatrix;

    EcalWeightSet();
    EcalWeightSet(const EcalWeightSet& aset);
    ~EcalWeightSet();

    EcalWeightMatrix& getWeightsBeforeGainSwitch() { return wgtBeforeSwitch_; }
    EcalWeightMatrix& getWeightsAfterGainSwitch()  { return wgtAfterSwitch_; }
    EcalWeightMatrix& getChi2WeightsBeforeGainSwitch()             { return wgtChi2BeforeSwitch_; }
    EcalWeightMatrix& getChi2WeightsAfterGainSwitch()             { return wgtChi2AfterSwitch_; }

    const EcalWeightMatrix& getWeightsBeforeGainSwitch() const { return wgtBeforeSwitch_; }
    const EcalWeightMatrix& getWeightsAfterGainSwitch()  const { return wgtAfterSwitch_; }
    const EcalWeightMatrix& getChi2WeightsBeforeGainSwitch()             const { return wgtChi2BeforeSwitch_; }
    const EcalWeightMatrix& getChi2WeightsAfterGainSwitch()             const { return wgtChi2AfterSwitch_; }

    EcalWeightSet& operator=(const EcalWeightSet& rhs);

    void print(std::ostream& o) const {
       using namespace std;
       o << "wgtBeforeSwitch_.size: " << wgtBeforeSwitch_.size()
            << " wgtAfterSwitch_.size: " << wgtAfterSwitch_.size()
            << " wgtChi2BeforeSwitch_.size: " << wgtChi2BeforeSwitch_.size()
            << " wgtChi2AfterSwitch_.size: " << wgtChi2AfterSwitch_.size()
            << endl;
    }


  private:
     std::vector< std::vector< EcalWeight > > wgtBeforeSwitch_;
     std::vector< std::vector< EcalWeight > > wgtAfterSwitch_;
     std::vector< std::vector< EcalWeight > > wgtChi2BeforeSwitch_;
     std::vector< std::vector< EcalWeight > > wgtChi2AfterSwitch_;
};

#endif
