#ifndef L1Trigger_Phase2L1ParticleFlow_newfirmware_hgcalinput_ref_h
#define L1Trigger_Phase2L1ParticleFlow_newfirmware_hgcalinput_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
#include "conifer.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class HgcalClusterDecoderEmulator {
  public:
    class MultiClassID {
    public:
      typedef ap_ufixed<9, 9, AP_RND_CONV, AP_SAT> bdt_feature_t;
      typedef ap_fixed<18, 4, AP_RND_CONV, AP_SAT> bdt_score_t;

      struct WPs {
        WPs(const std::vector<double> &wp_pt,
            const std::vector<double> &wp_PU,
            const std::vector<double> &wp_Pi,
            const std::vector<double> &wp_PFEm,
            const std::vector<double> &wp_EgEm,
            const std::vector<double> &wp_EgEm_tight);

        WPs(const edm::ParameterSet &pset);
        static edm::ParameterSetDescription getParameterSetDescription();

        std::vector<l1ct::pt_t> wp_pt;
        std::vector<l1ct::id_prob_t> wp_PU;
        std::vector<l1ct::id_prob_t> wp_Pi;
        std::vector<l1ct::id_prob_t> wp_PFEm;
        std::vector<l1ct::id_prob_t> wp_EgEm;
        std::vector<l1ct::id_prob_t> wp_EgEm_tight;
      };

      MultiClassID(const std::string &model,
                   const std::vector<double> &wp_eta,
                   const std::vector<l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs> &wps);
      MultiClassID(const edm::ParameterSet &pset);
      static edm::ParameterSetDescription getParameterSetDescription();

      bool evaluate(const l1ct::PFRegionEmu &sector,
                    l1ct::HadCaloObjEmu &cl,
                    const std::vector<bdt_feature_t> &input) const;

      void softmax(const float rawScores[3], float scores[3]) const;

    private:
      void initialize(const std::string &model, const std::vector<double> &wp_eta);

      std::vector<l1ct::glbeta_t> wp_eta_;
      std::vector<l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs> wps_;

      typedef ap_fixed<18, 8> activation_table_t;
      typedef ap_fixed<18, 8, AP_RND, AP_SAT> activation_exp_table_t;
      typedef ap_fixed<18, 8, AP_RND, AP_SAT> activation_inv_table_t;

      struct softmax_config {
        static const unsigned n_in = 3;
        static const unsigned table_size = 1024;
        typedef activation_exp_table_t exp_table_t;
        typedef activation_inv_table_t inv_table_t;
      };

      std::unique_ptr<conifer::BDT<bdt_feature_t, bdt_score_t, false>> multiclass_bdt_;
    };

    HgcalClusterDecoderEmulator(const std::string &model,
                                const std::vector<double> &wp_eta,
                                const std::vector<l1ct::HgcalClusterDecoderEmulator::MultiClassID::WPs> &wps,
                                bool slim = false,
                                const std::string &corrector = "",
                                float correctorEmfMax = -1,
                                bool emulateCorrections = false,
                                const std::string &emInterpScenario = "no");
    HgcalClusterDecoderEmulator(const edm::ParameterSet &pset);

    enum class UseEmInterp { No, EmOnly, AllKeepHad, AllKeepTot };

    ~HgcalClusterDecoderEmulator() = default;

    static edm::ParameterSetDescription getParameterSetDescription();

    l1ct::HadCaloObjEmu decode(const l1ct::PFRegionEmu &sector, const ap_uint<256> &in, bool &valid) const;

  private:
    UseEmInterp setEmInterpScenario(const std::string &emInterpScenario);
    bool slim_;
    l1ct::HgcalClusterDecoderEmulator::MultiClassID multiclass_id_;
    l1tpf::corrector corrector_;
    UseEmInterp emInterpScenario_;
  };
}  // namespace l1ct

#endif
