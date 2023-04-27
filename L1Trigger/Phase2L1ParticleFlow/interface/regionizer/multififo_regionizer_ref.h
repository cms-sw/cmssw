#ifndef multififo_regionizer_ref_h
#define multififo_regionizer_ref_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/regionizer_base_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_elements_ref.h"
#include <memory>

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class EGInputSelectorEmulator;
  struct EGInputSelectorEmuConfig;
}  // namespace l1ct

namespace l1ct {
  class MultififoRegionizerEmulator : public RegionizerEmulator {
  public:
    MultififoRegionizerEmulator(unsigned int nendcaps,
                                unsigned int nclocks,
                                unsigned int ntklinks,
                                unsigned int ncalolinks,
                                unsigned int ntk,
                                unsigned int ncalo,
                                unsigned int nem,
                                unsigned int nmu,
                                bool streaming,
                                unsigned int outii,
                                unsigned int pauseii,
                                bool useAlsoVtxCoords);

    enum class BarrelSetup { Full54, Full27, Central18, Central9, Phi18, Phi9 };
    MultififoRegionizerEmulator(BarrelSetup barrelSetup,
                                unsigned int nHCalLinks,
                                unsigned int nECalLinks,
                                unsigned int nclocks,
                                unsigned int ntk,
                                unsigned int ncalo,
                                unsigned int nem,
                                unsigned int nmu,
                                bool streaming,
                                unsigned int outii,
                                unsigned int pauseii,
                                bool useAlsoVtxCoords);
    // note: these ones will work only in CMSSW
    MultififoRegionizerEmulator(const edm::ParameterSet& iConfig);
    MultififoRegionizerEmulator(const std::string& barrelSetup, const edm::ParameterSet& iConfig);

    ~MultififoRegionizerEmulator() override;

    static BarrelSetup parseBarrelSetup(const std::string& setup);

    void setEgInterceptMode(bool afterFifo, const l1ct::EGInputSelectorEmuConfig& interceptorConfig);
    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

    // clock-cycle emulation
    bool step(bool newEvent,
              const std::vector<l1ct::TkObjEmu>& links,
              std::vector<l1ct::TkObjEmu>& out,
              bool mux = true);
    bool step(bool newEvent,
              const std::vector<l1ct::EmCaloObjEmu>& links,
              std::vector<l1ct::EmCaloObjEmu>& out,
              bool mux = true);
    bool step(bool newEvent,
              const std::vector<l1ct::HadCaloObjEmu>& links,
              std::vector<l1ct::HadCaloObjEmu>& out,
              bool mux = true);
    bool step(bool newEvent,
              const std::vector<l1ct::MuObjEmu>& links,
              std::vector<l1ct::MuObjEmu>& out,
              bool mux = true);
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
    void destream(int iclock,
                  const std::vector<l1ct::TkObjEmu>& tk_out,
                  const std::vector<l1ct::EmCaloObjEmu>& em_out,
                  const std::vector<l1ct::HadCaloObjEmu>& calo_out,
                  const std::vector<l1ct::MuObjEmu>& mu_out,
                  PFInputRegion& out);

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

    // convert links to firmware
    void toFirmware(const std::vector<l1ct::TkObjEmu>& emu, TkObj fw[/*NTK_SECTORS*NTK_LINKS*/]);
    void toFirmware(const std::vector<l1ct::HadCaloObjEmu>& emu, HadCaloObj fw[/*NCALO_SECTORS*NCALO_LINKS*/]);
    void toFirmware(const std::vector<l1ct::EmCaloObjEmu>& emu, EmCaloObj fw[/*NCALO_SECTORS*NCALO_LINKS*/]);
    void toFirmware(const std::vector<l1ct::MuObjEmu>& emu, MuObj fw[/*NMU_LINKS*/]);

    void reset();

  private:
    const unsigned int NTK_SECTORS, NCALO_SECTORS;  // max objects per sector per clock cycle
    const unsigned int NTK_LINKS, NCALO_LINKS, HCAL_LINKS, ECAL_LINKS, NMU_LINKS;
    unsigned int nendcaps_, nclocks_, ntk_, ncalo_, nem_, nmu_, outii_, pauseii_, nregions_;
    bool streaming_;
    enum EmInterceptMode { noIntercept = 0, interceptPreFifo, interceptPostFifo } emInterceptMode_;
    std::unique_ptr<EGInputSelectorEmulator> interceptor_;
    bool init_;

    multififo_regionizer::Regionizer<l1ct::TkObjEmu> tkRegionizer_;
    multififo_regionizer::Regionizer<l1ct::HadCaloObjEmu> hadCaloRegionizer_;
    multififo_regionizer::Regionizer<l1ct::EmCaloObjEmu> emCaloRegionizer_;
    multififo_regionizer::Regionizer<l1ct::MuObjEmu> muRegionizer_;
    std::vector<l1ct::multififo_regionizer::Route> tkRoutes_, caloRoutes_, emCaloRoutes_, muRoutes_;

    template <typename T>
    void fillCaloLinks(unsigned int iclock,
                       const std::vector<DetectorSector<T>>& in,
                       std::vector<T>& links,
                       std::vector<bool>& valid);
  };

}  // namespace l1ct

#endif
