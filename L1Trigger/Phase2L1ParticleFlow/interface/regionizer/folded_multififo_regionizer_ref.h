#ifndef folded_multififo_regionizer_ref_h
#define folded_multififo_regionizer_ref_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_ref.h"
#include <memory>
#include <deque>

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class EGInputSelectorEmulator;
  struct EGInputSelectorEmuConfig;
}  // namespace l1ct

namespace l1ct {
  class FoldedMultififoRegionizerEmulator : public RegionizerEmulator {
  public:
    enum class FoldMode { EndcapEta2 };

    FoldedMultififoRegionizerEmulator(unsigned int nclocks,
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

#if 0
    MultififoRegionizerEmulator(MultififoRegionizerEmulator::BarrelSetup barrelSetup,
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
#endif

    // note: this one will work only in CMSSW
    FoldedMultififoRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~FoldedMultififoRegionizerEmulator() override;

    void setEgInterceptMode(bool afterFifo, const l1ct::EGInputSelectorEmuConfig& interceptorConfig);
    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

// clock-cycle emulation
#if 0
    bool step(bool newEvent,
              const std::vector<l1ct::HadCaloObjEmu>& links,
              std::vector<l1ct::HadCaloObjEmu>& out,
              bool mux = true);
#endif
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

    // link emulation from decoded inputs (for simulation)
    void fillEvent(const RegionizerDecodedInputs& in);

    template <typename T>
    void fillLinks(unsigned int iclock, std::vector<T>& links, std::vector<bool>& valid) {
      Fold& fold = fold_[whichFold(iclock)];
      fold.regionizer->fillLinks(iclock % clocksPerFold_, fold.sectors, links, valid);
    }

    // convert links to firmware
    template <typename TEmu, typename TFw>
    void toFirmware(const std::vector<TEmu>& emu, TFw fw[]) {
      fold_.front().regionizer->toFirmware(emu, fw);
    }

  protected:
    const unsigned int NTK_SECTORS, NCALO_SECTORS;
    const unsigned int NTK_LINKS, NCALO_LINKS, HCAL_LINKS, ECAL_LINKS, NMU_LINKS;
    unsigned int nclocks_, ntk_, ncalo_, nem_, nmu_, outii_, pauseii_, nregions_;
    bool streaming_;
    FoldMode foldMode_;
    bool init_;
    struct Fold {
      unsigned int index;
      std::unique_ptr<l1ct::MultififoRegionizerEmulator> regionizer;
      RegionizerDecodedInputs sectors;
      std::vector<PFInputRegion> regions;
      Fold(unsigned int i, std::unique_ptr<l1ct::MultififoRegionizerEmulator>&& ptr)
          : index(i), regionizer(std::move(ptr)) {}
    };
    std::vector<Fold> fold_;
    unsigned int clocksPerFold_;
    unsigned int iclock_;

    unsigned int whichFold(const l1ct::PFRegion& reg);
    unsigned int whichFold(unsigned int iclock) { return (iclock % nclocks_) / clocksPerFold_; }
    bool inFold(const l1ct::PFRegion& reg, const Fold& fold);
    void splitSectors(const RegionizerDecodedInputs& in);
    void splitRegions(const std::vector<PFInputRegion>& out);
  };
}  // namespace l1ct

#endif
