#ifndef tdr_regionizer_ref_h
#define tdr_regionizer_ref_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/regionizer_base_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_regionizer_elements_ref.h"

// This is the second, alternate implementation of the TDR regionizer, which doesn't use a pipe for overlaps

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {
  class TDRRegionizerEmulator : public RegionizerEmulator {
  public:
    TDRRegionizerEmulator(uint32_t ntk,
                          uint32_t ncalo,
                          uint32_t nem,
                          uint32_t nmu,
                          bool debug_tk = false,
                          bool debug_calo = false,
                          bool debug_emcalo = false,
                          bool debug_mu = false);

    // note: this one will work only in CMSSW
    TDRRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~TDRRegionizerEmulator() override;

    static edm::ParameterSetDescription getParameterSetDescription();

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

  private:
    /// The maximum number of objects of each type to output per small region
    uint32_t ntk_, ncalo_, nem_, nmu_;

    bool init_;  // has initialization happened

    tdr_regionizer::Regionizer<l1ct::TkObjEmu> tkRegionizers_;
    tdr_regionizer::Regionizer<l1ct::HadCaloObjEmu> hadCaloRegionizers_;
    tdr_regionizer::Regionizer<l1ct::EmCaloObjEmu> emCaloRegionizers_;
    tdr_regionizer::Regionizer<l1ct::MuObjEmu> muRegionizers_;
  };

}  // namespace l1ct

#endif
