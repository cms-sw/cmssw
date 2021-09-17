#ifndef L1Trigger_Phase2L1ParticleFlow_BitwisePFAlgo_h
#define L1Trigger_Phase2L1ParticleFlow_BitwisePFAlgo_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgoBase.h"

struct pfalgo_config;

namespace l1tpf_impl {
  class BitwisePFAlgo : public PFAlgoBase {
  public:
    BitwisePFAlgo(const edm::ParameterSet&);
    ~BitwisePFAlgo() override;
    void runPF(Region& r) const override;

  protected:
    enum class AlgoChoice { algo3, algo2hgc } algo_;
    std::shared_ptr<pfalgo_config> config_;
  };

}  // namespace l1tpf_impl

#endif
