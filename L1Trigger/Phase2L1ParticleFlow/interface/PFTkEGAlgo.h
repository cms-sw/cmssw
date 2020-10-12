#ifndef L1Trigger_Phase2L1ParticleFlow_PFTkEGAlgo_h
#define L1Trigger_Phase2L1ParticleFlow_PFTkEGAlgo_h

#include <algorithm>
#include <vector>

#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace l1tpf_impl {

  class PFTkEGAlgo {
  public:
    PFTkEGAlgo(const edm::ParameterSet &);
    virtual ~PFTkEGAlgo();
    void runTkEG(Region &r) const;

    bool writeEgSta() const {
      return writeEgSta_;
    }

  protected:
    int debug_;
    bool doBremRecovery_;
    bool filterHwQuality_;
    int caloHwQual_;
    float dEtaMaxBrem_;
    float dPhiMaxBrem_;
    std::vector<double> absEtaBoundaries_;
    std::vector<double> dEtaValues_;
    std::vector<double> dPhiValues_;
    float caloEtMin_;
    float trkQualityPtMin_;
    float trkQualityChi2_;
    bool writeEgSta_;

    void initRegion(Region &r) const;
    void link_emCalo2emCalo(Region &r, std::vector<int> &emCalo2emCalo) const;
    void link_emCalo2tk(Region &r, std::vector<int> &emCalo2tk) const;

    void eg_algo(Region &r, const std::vector<int> &emCalo2emCalo, const std::vector<int> &emCalo2tk) const;

    l1tpf_impl::EgObjectIndexer &addEgObjsToPF(std::vector<l1tpf_impl::EgObjectIndexer> &egobjs,
                                               const int calo_idx,
                                               const int hwQual,
                                               const float ptCorr = -1,
                                               const int tk_idx = -1,
                                               const float iso = -1) const;
  };

}  // namespace l1tpf_impl

#endif
