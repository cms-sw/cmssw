#include "RecoLocalCalo/HcalRecAlgos/interface/AbsPlan1RechitCombiner.h"

float AbsPlan1RechitCombiner::energyWeightedAverage(const FPair* data,
                                                    const unsigned len,
                                                    const float valueToReturnOnFailure) {
  double sum = 0.0, wsum = 0.0;
  for (unsigned i = 0; i < len; ++i) {
    const float w = data[i].second;
    if (w > 0.f) {
      sum += w * data[i].first;
      wsum += w;
    }
  }
  if (wsum > 0.0)
    return sum / wsum;
  else
    return valueToReturnOnFailure;
}
