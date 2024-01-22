#ifndef middle_buffer_multififo_regionizer_ref_h
#define middle_buffer_multififo_regionizer_ref_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include <memory>
#include <deque>

namespace l1ct {
  class MiddleBufferMultififoRegionizerEmulator : public RegionizerEmulator {
  public:
    MiddleBufferMultififoRegionizerEmulator(unsigned int nclocks,
                                            unsigned int nbuffers,
                                            unsigned int etabufferDepth,
                                            unsigned int ntklinks,
                                            unsigned int nHCalLinks,
                                            unsigned int nECalLinks,
                                            unsigned int ntk,
                                            unsigned int ncalo,
                                            unsigned int nem,
                                            unsigned int nmu,
                                            bool streaming,
                                            unsigned int outii,
                                            unsigned int pauseii,
                                            bool useAlsoVtxCoords);
    // note: this one will work only in CMSSW
    MiddleBufferMultififoRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~MiddleBufferMultififoRegionizerEmulator() override;

    static edm::ParameterSetDescription getParameterSetDescription();

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

    // link emulation from decoded inputs (for simulation)
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::TkObjEmu>& links,
                   std::vector<bool>& valid);
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::HadCaloObjEmu>& links,
                   std::vector<bool>& valid);
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::EmCaloObjEmu>& links,
                   std::vector<bool>& valid);
    void fillLinks(unsigned int iclock,
                   const RegionizerDecodedInputs& in,
                   std::vector<l1ct::MuObjEmu>& links,
                   std::vector<bool>& valid);
    template <typename T>
    void fillLinks(unsigned int iclock, const RegionizerDecodedInputs& in, std::vector<T>& links) {
      std::vector<bool> unused;
      fillLinks(iclock, in, links, unused);
    }

    void destream(int iclock,
                  const std::vector<l1ct::TkObjEmu>& tk_out,
                  const std::vector<l1ct::EmCaloObjEmu>& em_out,
                  const std::vector<l1ct::HadCaloObjEmu>& calo_out,
                  const std::vector<l1ct::MuObjEmu>& mu_out,
                  PFInputRegion& out);

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
              bool /*unused*/);

    template <typename TEmu, typename TFw>
    void toFirmware(const std::vector<TEmu>& emu, TFw fw[]) {
      for (unsigned int i = 0, n = emu.size(); i < n; ++i) {
        fw[i] = emu[i];
      }
    }

    void reset();

  protected:
    const unsigned int NTK_SECTORS, NCALO_SECTORS;
    const unsigned int NTK_LINKS, HCAL_LINKS, ECAL_LINKS, NMU_LINKS;
    unsigned int nclocks_, nbuffers_, etabuffer_depth_, ntk_, ncalo_, nem_, nmu_, outii_, pauseii_, nregions_pre_,
        nregions_post_;
    bool streaming_;
    bool init_;
    unsigned int iclock_;
    std::vector<l1ct::PFRegionEmu> mergedRegions_, outputRegions_;
    multififo_regionizer::Regionizer<l1ct::TkObjEmu> tkRegionizerPre_, tkRegionizerPost_;
    multififo_regionizer::Regionizer<l1ct::HadCaloObjEmu> hadCaloRegionizerPre_, hadCaloRegionizerPost_;
    multififo_regionizer::Regionizer<l1ct::EmCaloObjEmu> emCaloRegionizerPre_, emCaloRegionizerPost_;
    multififo_regionizer::Regionizer<l1ct::MuObjEmu> muRegionizerPre_, muRegionizerPost_;
    std::vector<l1ct::multififo_regionizer::Route> tkRoutes_, caloRoutes_, emCaloRoutes_, muRoutes_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::TkObjEmu>> tkBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::HadCaloObjEmu>> hadCaloBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::EmCaloObjEmu>> emCaloBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::MuObjEmu>> muBuffers_;

    template <typename T>
    void fillCaloLinks_(unsigned int iclock,
                        const std::vector<DetectorSector<T>>& in,
                        std::vector<T>& links,
                        std::vector<bool>& valid);

    void fillSharedCaloLinks(unsigned int iclock,
                             const std::vector<DetectorSector<l1ct::EmCaloObjEmu>>& em_in,
                             const std::vector<DetectorSector<l1ct::HadCaloObjEmu>>& had_in,
                             std::vector<l1ct::HadCaloObjEmu>& links,
                             std::vector<bool>& valid);

    void encode(const l1ct::EmCaloObjEmu& from, l1ct::HadCaloObjEmu& to);
    void encode(const l1ct::HadCaloObjEmu& from, l1ct::HadCaloObjEmu& to);
    void decode(l1ct::HadCaloObjEmu& had, l1ct::EmCaloObjEmu& em);
  };
}  // namespace l1ct

#endif
