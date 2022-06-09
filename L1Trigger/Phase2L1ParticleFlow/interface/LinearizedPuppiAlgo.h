#ifndef L1Trigger_Phase2L1ParticleFlow_LinearizedPuppiAlgo_h
#define L1Trigger_Phase2L1ParticleFlow_LinearizedPuppiAlgo_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/PuppiAlgo.h"

namespace l1tpf_impl {

  class LinearizedPuppiAlgo : public PuppiAlgo {
  public:
    LinearizedPuppiAlgo(const edm::ParameterSet &);
    ~LinearizedPuppiAlgo() override;

    const std::vector<std::string> &puGlobalNames() const override;
    void doPUGlobals(const std::vector<Region> &rs, float z0, float npu, std::vector<float> &globals) const override;
    void runNeutralsPU(Region &r, float z0, float npu, const std::vector<float> &globals) const override;

  protected:
    void computePuppiWeights(Region &r,
                             float npu,
                             const std::vector<float> &alphaC,
                             const std::vector<float> &alphaF) const;

    std::vector<float> puppiPriors_, puppiPriorsPhotons_;
    std::vector<float> puppiPtSlopes_, puppiPtSlopesPhotons_;
    std::vector<float> puppiPtZeros_, puppiPtZerosPhotons_;
    std::vector<float> puppiAlphaSlopes_, puppiAlphaSlopesPhotons_;
    std::vector<float> puppiAlphaZeros_, puppiAlphaZerosPhotons_;
    std::vector<float> puppiAlphaCrops_, puppiAlphaCropsPhotons_;
  };

}  // namespace l1tpf_impl

#endif
