#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOW_BUFFERED_FOLDED_MULTIFIFO_REGIONZER_REF_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOW_BUFFERED_FOLDED_MULTIFIFO_REGIONZER_REF_H

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/folded_multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include <memory>
#include <deque>

namespace l1ct {
  class BufferedFoldedMultififoRegionizerEmulator : public FoldedMultififoRegionizerEmulator {
  public:
    enum class FoldMode { EndcapEta2 };

    BufferedFoldedMultififoRegionizerEmulator(unsigned int nclocks,
                                              unsigned int ntk,
                                              unsigned int ncalo,
                                              unsigned int nem,
                                              unsigned int nmu,
                                              bool streaming,
                                              unsigned int outii,
                                              unsigned int pauseii,
                                              bool useAlsoVtxCoords);
    // note: this one will work only in CMSSW
    BufferedFoldedMultififoRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~BufferedFoldedMultififoRegionizerEmulator() override;

    static edm::ParameterSetDescription getParameterSetDescription();

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

    void fillLinks(unsigned int iclock, std::vector<l1ct::TkObjEmu>& links, std::vector<bool>& valid);
    void fillLinks(unsigned int iclock, std::vector<l1ct::HadCaloObjEmu>& links, std::vector<bool>& valid);
    void fillLinks(unsigned int iclock, std::vector<l1ct::EmCaloObjEmu>& links, std::vector<bool>& valid);
    void fillLinks(unsigned int iclock, std::vector<l1ct::MuObjEmu>& links, std::vector<bool>& valid);

    // clock-cycle emulation
    bool step(bool newEvent,
              const std::vector<l1ct::TkObjEmu>& links_tk,
              const std::vector<l1ct::HadCaloObjEmu>& links_hadCalo,
              const std::vector<l1ct::EmCaloObjEmu>& links_emCalo,
              const std::vector<l1ct::MuObjEmu>& links_mu,
              std::vector<l1ct::TkObjEmu>& out_tk,
              std::vector<l1ct::HadCaloObjEmu>& out_hadCalo,
              std::vector<l1ct::EmCaloObjEmu>& out_emCalo,
              std::vector<l1ct::MuObjEmu>& out_mu,
              bool mux = true);

    template <typename TEmu, typename TFw>
    void toFirmware(const std::vector<TEmu>& emu, TFw fw[]) {
      for (unsigned int i = 0, n = emu.size(); i < n; ++i) {
        fw[i] = emu[i];
      }
    }

  protected:
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::TkObjEmu>> tkBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::HadCaloObjEmu>> caloBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::MuObjEmu>> muBuffers_;

    void findEtaBounds_(const l1ct::PFRegionEmu& sec,
                        const std::vector<PFInputRegion>& reg,
                        l1ct::glbeta_t& etaMin,
                        l1ct::glbeta_t& etaMax);

    template <typename T>
    void fillLinksPosNeg_(unsigned int iclock,
                          const std::vector<l1ct::DetectorSector<T>>& secNeg,
                          const std::vector<l1ct::DetectorSector<T>>& secPos,
                          std::vector<T>& links,
                          std::vector<bool>& valid);
  };
}  // namespace l1ct

#endif
