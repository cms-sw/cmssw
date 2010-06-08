#ifndef _TP_SIDE_BAND_SUBTRACTION_
#define _TP_SIDE_BAND_SUBTRACTION_

#include <vector>

namespace edm {
  class ParameterSet;
}

class TH1F;

class SideBandSubtraction {
public:
  void Configure(const edm::ParameterSet& SBSPSet);
  void Subtract(const TH1F& Total, TH1F& Result);
private:
  double Peak_, SD_;
};

#endif
