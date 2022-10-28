#ifndef PFALGO2HGC_REF_H
#define PFALGO2HGC_REF_H

#include "pfalgo_common_ref.h"

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class PFAlgo2HGCEmulator : public PFAlgoEmulatorBase {
  public:
    PFAlgo2HGCEmulator(unsigned int nTrack,
                       unsigned int nCalo,
                       unsigned int nMu,
                       unsigned int nSelCalo,
                       unsigned int dR2Max_Tk_Mu,
                       unsigned int dR2Max_Tk_Calo,
                       pt_t tk_MaxInvPt_Loose,
                       pt_t tk_MaxInvPt_Tight)
        : PFAlgoEmulatorBase(
              nTrack, nCalo, nMu, nSelCalo, dR2Max_Tk_Mu, dR2Max_Tk_Calo, tk_MaxInvPt_Loose, tk_MaxInvPt_Tight) {}

    // note: this one will work only in CMSSW
    PFAlgo2HGCEmulator(const edm::ParameterSet& iConfig);

    ~PFAlgo2HGCEmulator() override {}

    void run(const PFInputRegion& in, OutputRegion& out) const override;

    /// moves all objects from out.pfphoton to the beginning of out.pfneutral: nothing to do for this algo
    void mergeNeutrals(OutputRegion& out) const override {}

    void toFirmware(const PFInputRegion& in,
                    PFRegion& region,
                    HadCaloObj calo[/*nCALO*/],
                    TkObj track[/*nTRACK*/],
                    MuObj mu[/*nMU*/]) const;
    void toFirmware(const OutputRegion& out,
                    PFChargedObj outch[/*nTRACK*/],
                    PFNeutralObj outne[/*nSELCALO*/],
                    PFChargedObj outmu[/*nMU*/]) const;
  };

}  // namespace l1ct

#endif
