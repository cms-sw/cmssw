#ifndef __L1Analysis_L1AnalysisRecoPhotonDataFormat_H__
#define __L1Analysis_L1AnalysisRecoPhotonDataFormat_H__

//-------------------------------------------------------------------------------
// Original code : L1Trigger/L1TNtuples/L1RecoPhotonNtupleProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis {
  struct L1AnalysisRecoPhotonDataFormat {
    L1AnalysisRecoPhotonDataFormat() { Reset(); };
    ~L1AnalysisRecoPhotonDataFormat() { Reset(); };

    void Reset() {
      nPhotons = 0;

      e.clear();
      et.clear();
      pt.clear();
      eta.clear();
      phi.clear();
      r9.clear();
      hasPixelSeed.clear();
      isTightPhoton.clear();
      isLoosePhoton.clear();
    }

    unsigned nPhotons;
    std::vector<float> e;
    std::vector<float> et;
    std::vector<float> pt;
    std::vector<float> eta;
    std::vector<float> phi;
    std::vector<float> r9;
    std::vector<short> hasPixelSeed;
    std::vector<short> isTightPhoton;
    std::vector<short> isLoosePhoton;
  };
}  // namespace L1Analysis
#endif
