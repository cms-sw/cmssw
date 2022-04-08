#ifndef L1Trigger_Phase2L1ParticleFlow_PUAlgoBase_h
#define L1Trigger_Phase2L1ParticleFlow_PUAlgoBase_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace l1tpf_impl {

  class PUAlgoBase {
  public:
    PUAlgoBase(const edm::ParameterSet &);
    virtual ~PUAlgoBase();

    /// global operations
    enum class VertexAlgo { Old, TP, External };
    virtual void doVertexing(std::vector<Region> &rs,
                             VertexAlgo algo,
                             float &vz) const;  // region is not const since it sets the fromPV bit of the tracks

    virtual void doVertexings(
        std::vector<Region> &rs,
        VertexAlgo algo,
        std::vector<float> &vz) const;  // region is not const since it sets the fromPV bit of the tracks

    virtual void runChargedPV(Region &r, float z0) const;
    virtual void runChargedPV(Region &r, std::vector<float> &z0) const;

    virtual const std::vector<std::string> &puGlobalNames() const;
    virtual void doPUGlobals(const std::vector<Region> &rs, float z0, float npu, std::vector<float> &globals) const = 0;
    virtual void runNeutralsPU(Region &r, float z0, float npu, const std::vector<float> &globals) const = 0;
    virtual void runNeutralsPU(Region &r,
                               std::vector<float> &z0,
                               float npu,
                               const std::vector<float> &globals) const = 0;

  protected:
    int debug_;
    float etaCharged_, vtxRes_;
    bool vtxAdaptiveCut_;
    int nVtx_;
  };

}  // namespace l1tpf_impl

#endif
