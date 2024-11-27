#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_hgcalinput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_hgcalinput_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
// FIXME: move to the include from external package
#include "L1Trigger/Phase2L1ParticleFlow/interface/conifer.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class HgcalClusterDecoderEmulator {
  public:
    HgcalClusterDecoderEmulator(const std::string &model,
                                const std::vector<double> &wp_pt,
                                const std::vector<double> &wp_PU,
                                const std::vector<double> &wp_Pi,
                                const std::vector<double> &wp_EgEm,
                                const std::vector<double> &wp_PFEm,
                                bool slim = false);
    HgcalClusterDecoderEmulator(const edm::ParameterSet &pset);

    class MultiClassID {
    public:
      typedef ap_fixed<20, 10> bdt_feature_t;
      typedef ap_fixed<20, 6> bdt_score_t;

      MultiClassID(const std::string &model,
                   const std::vector<double> &wp_pt,
                   const std::vector<double> &wp_PU,
                   const std::vector<double> &wp_Pi,
                   const std::vector<double> &wp_EgEm,
                   const std::vector<double> &wp_PFEm);
      MultiClassID(const edm::ParameterSet &pset);
      static edm::ParameterSetDescription getParameterSetDescription();

      bool evaluate(l1ct::HadCaloObjEmu &cl, const std::vector<bdt_feature_t> &input) const;

      void softmax(const float rawScores[3], float scores[3]) const;

    private:
      std::vector<double> wp_pt_;
      std::vector<double> wp_PU_;
      std::vector<double> wp_Pi_;
      std::vector<double> wp_EgEm_;
      std::vector<double> wp_PFEm_;

      conifer::BDT<bdt_feature_t, bdt_score_t, false> *multiclass_bdt_;
    };

    ~HgcalClusterDecoderEmulator();

    static edm::ParameterSetDescription getParameterSetDescription();

    l1ct::HadCaloObjEmu decode(const l1ct::PFRegionEmu &sector, const ap_uint<256> &in, bool &valid) const;

  private:
    bool slim_;
    l1ct::HgcalClusterDecoderEmulator::MultiClassID multiclass_id_;
    l1tpf::corrector corrector_;  // FIXME: need to work outside of CMSSW as well: emulator version
  };
}  // namespace l1ct

#endif
