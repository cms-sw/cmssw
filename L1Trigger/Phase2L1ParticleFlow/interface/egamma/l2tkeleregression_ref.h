#ifndef L2TKELEREGRESSION_REF_H
#define L2TKELEREGRESSION_REF_H

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "DataFormats/L1TParticleFlow/interface/egamma.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <vector>
#include "conifer.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class L2TkEleRegressionEmulator {
  public:
    enum class ModelType { null = 0, EB_v0 = 1 };

    L2TkEleRegressionEmulator(const std::vector<double>& eta_bins,
                              const std::vector<unsigned int>& model_types,
                              const std::vector<std::string>& model_paths,
                              int debug);

    L2TkEleRegressionEmulator(const edm::ParameterSet& iConfig);
    static edm::ParameterSetDescription getParameterSetDescription();

    class Model {
    public:
      Model() = default;
      virtual ~Model() = default;

      virtual pt_t compute_ptCorr(const EGIsoEleObjEmu& ele) const = 0;
    };

    class Model_EB_v0 : public Model {
    public:
      Model_EB_v0(const std::string& model_path, int debug);

      l1ct::pt_t compute_ptCorr(const EGIsoEleObjEmu& ele) const override;

      typedef ap_fixed<10, 1, AP_RND_CONV, AP_SAT> bdt_feature_t;
      typedef ap_fixed<12, 3, AP_RND_CONV, AP_SAT> bdt_out_t;

    private:
      std::unique_ptr<conifer::BDT<bdt_feature_t, bdt_out_t, false>> model_;
      bdt_feature_t scale(float x, float min_x, int bitshift, float inf = -1) const {
        const float delta = x - min_x;
        if (bitshift >= 0) {
          const float denom = float(1u << unsigned(bitshift));
          return bdt_feature_t(inf + delta / denom);
        }
        const float mul = float(1u << unsigned(-bitshift));
        return bdt_feature_t(inf + delta * mul);
      }
    };

    void toFirmware(const std::vector<ap_uint<64>>& encoded_in, ap_uint<64> encoded_fw[]) const;

    void run(const std::vector<EGIsoEleObjEmu>& in_eles, std::vector<EGIsoEleObjEmu>& out_eles) const;

  private:
    std::vector<double> eta_bins_;
    std::vector<std::unique_ptr<Model>> models_;
  };

}  // namespace l1ct

#endif
