#ifndef CondFormats_ESObjects_ESWeightSet_HH
#define CondFormats_ESObjects_ESWeightSet_HH

#include "CondFormats/ESObjects/interface/ESWeight.h"
#include "DataFormats/Math/interface/Matrix.h"
#include <iostream>

class ESWeightSet {

  public:
  
  typedef math::Matrix<2,3>::type ESWeightMatrix;
  
  ESWeightSet();
  ESWeightSet(const ESWeightSet& aset);
  ESWeightSet(ESWeightMatrix& amat);
  ~ESWeightSet();
  
  ESWeightMatrix& getWeights() { return wgtBeforeSwitch_; }
  
  const ESWeightMatrix& getWeights() const { return wgtBeforeSwitch_; }
  
  ESWeightSet& operator=(const ESWeightSet& rhs);
  
  void print(std::ostream& o) const {
    using namespace std;
    o << "wgtBeforeSwitch_.: " << wgtBeforeSwitch_
      << endl;
  }
  
  
 private:
  ESWeightMatrix wgtBeforeSwitch_;
};

#endif
