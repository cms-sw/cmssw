#ifndef multififo_regionizer_ref_h
#define multififo_regionizer_ref_h

#include "../common/regionizer_base_ref.h"

#include "multififo_regionizer_elements_ref.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class MultififoRegionizerEmulator : public RegionizerEmulator {
  public:
    MultififoRegionizerEmulator(unsigned int nendcaps,
                                unsigned int nclocks,
                                unsigned int ntk,
                                unsigned int ncalo,
                                unsigned int nem,
                                unsigned int nmu,
                                bool streaming,
                                unsigned int outii = 0);

    // note: this one will work only in CMSSW
    MultififoRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~MultififoRegionizerEmulator() override;

    static const int NTK_SECTORS = 9, NTK_LINKS = 2;  // max objects per sector per clock cycle
    static const int NCALO_SECTORS = 3, NCALO_LINKS = 4;
    static const int NMU_LINKS = 2;

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    // TODO: implement
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
    void destream(int iclock,
                  const std::vector<l1ct::TkObjEmu>& tk_out,
                  const std::vector<l1ct::EmCaloObjEmu>& em_out,
                  const std::vector<l1ct::HadCaloObjEmu>& calo_out,
                  const std::vector<l1ct::MuObjEmu>& mu_out,
                  PFInputRegion& out);

    // link emulation from decoded inputs (for simulation)
    void fillLinks(unsigned int iclock, const RegionizerDecodedInputs& in, std::vector<l1ct::TkObjEmu>& links);
    void fillLinks(unsigned int iclock, const RegionizerDecodedInputs& in, std::vector<l1ct::HadCaloObjEmu>& links);
    void fillLinks(unsigned int iclock, const RegionizerDecodedInputs& in, std::vector<l1ct::EmCaloObjEmu>& links);
    void fillLinks(unsigned int iclock, const RegionizerDecodedInputs& in, std::vector<l1ct::MuObjEmu>& links);

    // convert links to firmware
    void toFirmware(const std::vector<l1ct::TkObjEmu>& emu, TkObj fw[NTK_SECTORS][NTK_LINKS]);
    void toFirmware(const std::vector<l1ct::HadCaloObjEmu>& emu, HadCaloObj fw[NCALO_SECTORS][NCALO_LINKS]);
    void toFirmware(const std::vector<l1ct::EmCaloObjEmu>& emu, EmCaloObj fw[NCALO_SECTORS][NCALO_LINKS]);
    void toFirmware(const std::vector<l1ct::MuObjEmu>& emu, MuObj fw[NMU_LINKS]);

  private:
    unsigned int nendcaps_, nclocks_, ntk_, ncalo_, nem_, nmu_, outii_, nregions_;
    bool streaming_;
    bool init_;

    multififo_regionizer::Regionizer<l1ct::TkObjEmu> tkRegionizer_;
    multififo_regionizer::Regionizer<l1ct::HadCaloObjEmu> hadCaloRegionizer_;
    multififo_regionizer::Regionizer<l1ct::EmCaloObjEmu> emCaloRegionizer_;
    multififo_regionizer::Regionizer<l1ct::MuObjEmu> muRegionizer_;
    std::vector<l1ct::multififo_regionizer::Route> tkRoutes_, caloRoutes_, muRoutes_;

    template <typename T>
    void fillCaloLinks_(unsigned int iclock, const std::vector<DetectorSector<T>>& in, std::vector<T>& links);
  };

}  // namespace l1ct

#endif
