#ifndef PFALGO_COMMON_REF_H
#define PFALGO_COMMON_REF_H

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#ifdef CMSSW_GIT_HASH
#include "pfalgo_types.h"
#else
#include "../firmware/pfalgo_types.h"
#endif

#include <algorithm>
#include <vector>

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class PFAlgoEmulatorBase {
  public:
    PFAlgoEmulatorBase(unsigned int nTrack,
                       unsigned int nCalo,
                       unsigned int nMu,
                       unsigned int nSelCalo,
                       unsigned int dR2Max_Tk_Mu,
                       unsigned int dR2Max_Tk_Calo,
                       pt_t tk_MaxInvPt_Loose,
                       pt_t tk_MaxInvPt_Tight)
        : nTRACK_(nTrack),
          nCALO_(nCalo),
          nMU_(nMu),
          nSELCALO_(nSelCalo),
          dR2MAX_TK_MU_(dR2Max_Tk_Mu),
          dR2MAX_TK_CALO_(dR2Max_Tk_Calo),
          tk_MAXINVPT_LOOSE_(tk_MaxInvPt_Loose),
          tk_MAXINVPT_TIGHT_(tk_MaxInvPt_Tight),
          debug_(false) {}

    virtual ~PFAlgoEmulatorBase();

    void loadPtErrBins(
        unsigned int nbins, const float absetas[], const float scales[], const float offs[], bool verbose = false);
    void loadPtErrBins(const edm::ParameterSet& iConfig);

    void setDebug(bool debug = true) { debug_ = debug; }

    virtual void run(const PFInputRegion& in, OutputRegion& out) const = 0;

    /// moves all objects from out.pfphoton to the beginning of out.pfneutral
    virtual void mergeNeutrals(OutputRegion& out) const = 0;

  protected:
    // config
    unsigned int nTRACK_, nCALO_, nMU_;
    unsigned int nSELCALO_;
    unsigned int dR2MAX_TK_MU_;
    unsigned int dR2MAX_TK_CALO_;
    pt_t tk_MAXINVPT_LOOSE_, tk_MAXINVPT_TIGHT_;

    struct ptErrBin {
      glbeta_t abseta;
      ptErrScale_t scale;
      ptErrOffs_t offs;
    };
    std::vector<ptErrBin> ptErrBins_;

    bool debug_;

    // tools
    template <typename COV>
    int best_match_with_pt_ref(int dR2MAX, const COV& calo, const TkObjEmu& track, const pt_t& trackCaloPtErr) const;

    template <typename TV>
    void ptsort_ref(int nIn, int nOut, const TV& in, TV& out) const;

    pt_t ptErr_ref(const PFRegionEmu& region, const TkObjEmu& track) const;

    void pfalgo_mu_ref(const PFInputRegion& in, OutputRegion& out, std::vector<int>& iMu) const;

    void fillPFCand(const TkObjEmu& track, PFChargedObjEmu& pf, bool isMu, bool isEle) const;
    void fillPFCand(const HadCaloObjEmu& calo, PFNeutralObjEmu& pf, bool isPhoton = false) const;
    void fillPFCand(const EmCaloObjEmu& calo, PFNeutralObjEmu& pf, bool isPhoton = true) const;
  };

}  // namespace l1ct
//=== begin implementation part

template <typename COV>
int l1ct::PFAlgoEmulatorBase::best_match_with_pt_ref(int dR2MAX,
                                                     const COV& calo,
                                                     const TkObjEmu& track,
                                                     const pt_t& trackCaloPtErr) const {
  pt_t caloPtMin = track.hwPt - 2 * trackCaloPtErr;
  if (caloPtMin < 0)
    caloPtMin = 0;
  float ptErr = std::max<float>(Scales::INTPT_LSB, Scales::floatPt(trackCaloPtErr));
  ptscale_t dptscale = float(dR2MAX) / (ptErr * ptErr);
  int dr2min = 0, ibest = -1;
  for (int ic = 0, nCAL = calo.size(); ic < nCAL; ++ic) {
    if (calo[ic].hwPt <= caloPtMin)
      continue;
    int dr2 = dr2_int(track.hwEta, track.hwPhi, calo[ic].hwEta, calo[ic].hwPhi);
    if (dr2 >= dR2MAX)
      continue;
    pt_t dpt = track.hwPt - calo[ic].hwPt;
    dr2 += int((dpt * dpt) * dptscale);
    if (ibest == -1 || dr2 < dr2min) {
      dr2min = dr2;
      ibest = ic;
    }
  }
  return ibest;
}

template <typename TV>
void l1ct::PFAlgoEmulatorBase::ptsort_ref(int nIn, int nOut, const TV& in, TV& out) const {
  out.resize(nOut);
  for (int iout = 0; iout < nOut; ++iout) {
    out[iout].clear();
  }
  for (int it = 0; it < nIn; ++it) {
    for (int iout = 0; iout < nOut; ++iout) {
      if (in[it].hwPt >= out[iout].hwPt) {
        for (int i2 = nOut - 1; i2 > iout; --i2) {
          out[i2] = out[i2 - 1];
        }
        out[iout] = in[it];
        break;
      }
    }
  }
}

#endif
