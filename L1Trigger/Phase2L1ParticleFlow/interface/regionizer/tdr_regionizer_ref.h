#ifndef tdr_regionizer_ref_h
#define tdr_regionizer_ref_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/regionizer_base_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_regionizer_elements_ref.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {
  class TDRRegionizerEmulator : public RegionizerEmulator {
  public:
    TDRRegionizerEmulator(unsigned int netaslices,
                          unsigned int ntk,
                          unsigned int ncalo,
                          unsigned int nem,
                          unsigned int nmu,
                          int nclocks,
                          bool dosort);

    // note: this one will work only in CMSSW
    TDRRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~TDRRegionizerEmulator() override;

    static const int NTK_SECTORS = 9, NTK_LINKS = 2;  // max objects per sector per clock cycle
    static const int NCALO_SECTORS = 4, NCALO_LINKS = 4;
    static const int NEMCALO_SECTORS = 4, NEMCALO_LINKS = 4;
    static const int NMU_LINKS = 2;
    static const int MAX_TK_EVT = 108, MAX_EMCALO_EVT = 162, MAX_CALO_EVT = 162,
                     MAX_MU_EVT = 162;  //all at TMUX 6, per link
    //assuming 96b for tracks, 64b for emcalo, calo, mu
    static const int NUMBER_OF_SMALL_REGIONS = 18;
    static const int NETA_SMALL = 2;

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    // TODO: implement
    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

    // link emulation from decoded inputs (for simulation)
    void fillLinks(const RegionizerDecodedInputs& in, std::vector<std::vector<l1ct::TkObjEmu>>& links);
    void fillLinks(const RegionizerDecodedInputs& in, std::vector<std::vector<l1ct::HadCaloObjEmu>>& links);
    void fillLinks(const RegionizerDecodedInputs& in, std::vector<std::vector<l1ct::EmCaloObjEmu>>& links);
    void fillLinks(const RegionizerDecodedInputs& in, std::vector<std::vector<l1ct::MuObjEmu>>& links);

    // convert links to firmware
    void toFirmware(const std::vector<l1ct::TkObjEmu>& emu, TkObj fw[NTK_SECTORS][NTK_LINKS]);
    void toFirmware(const std::vector<l1ct::HadCaloObjEmu>& emu, HadCaloObj fw[NCALO_SECTORS][NCALO_LINKS]);
    void toFirmware(const std::vector<l1ct::EmCaloObjEmu>& emu, EmCaloObj fw[NCALO_SECTORS][NCALO_LINKS]);
    void toFirmware(const std::vector<l1ct::MuObjEmu>& emu, MuObj fw[NMU_LINKS]);

  private:
    unsigned int netaslices_, ntk_, ncalo_, nem_, nmu_, nregions_;
    int nclocks_;
    bool dosort_, init_;

    std::vector<tdr_regionizer::Regionizer<l1ct::TkObjEmu>> tkRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::HadCaloObjEmu>> hadCaloRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::EmCaloObjEmu>> emCaloRegionizers_;
    std::vector<tdr_regionizer::Regionizer<l1ct::MuObjEmu>> muRegionizers_;
  };

}  // namespace l1ct

#endif
