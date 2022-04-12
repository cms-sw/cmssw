#ifndef PFALGO3_REF_H
#define PFALGO3_REF_H

#include "pfalgo_common_ref.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class PFAlgo3Emulator : public PFAlgoEmulatorBase {
  public:
    PFAlgo3Emulator(unsigned int nTrack,
                    unsigned int nEmCalo,
                    unsigned int nCalo,
                    unsigned int nMu,
                    unsigned int nPhoton,
                    unsigned int nSelCalo,
                    unsigned int nAllNeutral,
                    unsigned int dR2Max_Tk_Mu,
                    unsigned int dR2Max_Tk_Em,
                    unsigned int dR2Max_Em_Calo,
                    unsigned int dR2Max_Tk_Calo,
                    pt_t tk_MaxInvPt_Loose,
                    pt_t tk_MaxInvPt_Tight)
        : PFAlgoEmulatorBase(
              nTrack, nCalo, nMu, nSelCalo, dR2Max_Tk_Mu, dR2Max_Tk_Calo, tk_MaxInvPt_Loose, tk_MaxInvPt_Tight),
          nEMCALO_(nEmCalo),
          nPHOTON_(nPhoton),
          nALLNEUTRAL_(nAllNeutral),
          dR2MAX_TK_EM_(dR2Max_Tk_Em),
          dR2MAX_EM_CALO_(dR2Max_Em_Calo) {}

    // note: this one will work only in CMSSW
    PFAlgo3Emulator(const edm::ParameterSet& iConfig);

    ~PFAlgo3Emulator() override {}

    void run(const PFInputRegion& in, OutputRegion& out) const override;

    void toFirmware(const PFInputRegion& in,
                    PFRegion& region,
                    HadCaloObj calo[/*nCALO*/],
                    EmCaloObj emcalo[/*nEMCALO*/],
                    TkObj track[/*nTRACK*/],
                    MuObj mu[/*nMU*/]) const;
    void toFirmware(const OutputRegion& out,
                    PFChargedObj outch[/*nTRACK*/],
                    PFNeutralObj outpho[/*nPHOTON*/],
                    PFNeutralObj outne[/*nSELCALO*/],
                    PFChargedObj outmu[/*nMU*/]) const;

    /// moves all objects from out.pfphoton to the beginning of out.pfneutral
    void mergeNeutrals(OutputRegion& out) const override;

  protected:
    unsigned int nEMCALO_, nPHOTON_, nALLNEUTRAL_;
    unsigned int dR2MAX_TK_EM_;
    unsigned int dR2MAX_EM_CALO_;

    int tk_best_match_ref(unsigned int dR2MAX,
                          const std::vector<l1ct::EmCaloObjEmu>& calo,
                          const l1ct::TkObjEmu& track) const;
    int em_best_match_ref(unsigned int dR2MAX,
                          const std::vector<l1ct::HadCaloObjEmu>& calo,
                          const l1ct::EmCaloObjEmu& em) const;

    void pfalgo3_em_ref(const PFInputRegion& in,
                        const std::vector<int>& iMu /*[nTRACK]*/,
                        std::vector<int>& iEle /*[nTRACK]*/,
                        OutputRegion& out,
                        std::vector<HadCaloObjEmu>& hadcalo_out /*[nCALO]*/) const;
  };

}  // namespace l1ct

#endif
