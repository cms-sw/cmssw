#ifndef L1Trigger_Phase2L1ParticleFlow_PFALGO_COMMON_REF_H
#define L1Trigger_Phase2L1ParticleFlow_PFALGO_COMMON_REF_H

#include "../firmware/data.h"
#include "../firmware/pfalgo_common.h"
#include <algorithm>

template <typename T>
inline int sqr(const T &t) {
  return t * t;
}

template <typename CO_t>
int best_match_with_pt_ref(int nCAL, int dR2MAX, const CO_t calo[/*nCAL*/], const TkObj &track);

template <typename T>
void ptsort_ref(int nIn, int nOut, const T in[/*nIn*/], T out[/*nOut*/]);

struct pfalgo_config {
  unsigned int nTRACK, nCALO, nMU;
  unsigned int nSELCALO;
  unsigned int dR2MAX_TK_MU;
  unsigned int dR2MAX_TK_CALO;
  unsigned int tk_MAXINVPT_LOOSE, tk_MAXINVPT_TIGHT;

  pfalgo_config(unsigned int nTrack,
                unsigned int nCalo,
                unsigned int nMu,
                unsigned int nSelCalo,
                unsigned int dR2Max_Tk_Mu,
                unsigned int dR2Max_Tk_Calo,
                unsigned int tk_MaxInvPt_Loose,
                unsigned int tk_MaxInvPt_Tight)
      : nTRACK(nTrack),
        nCALO(nCalo),
        nMU(nMu),
        nSELCALO(nSelCalo),
        dR2MAX_TK_MU(dR2Max_Tk_Mu),
        dR2MAX_TK_CALO(dR2Max_Tk_Calo),
        tk_MAXINVPT_LOOSE(tk_MaxInvPt_Loose),
        tk_MAXINVPT_TIGHT(tk_MaxInvPt_Tight) {}

  virtual ~pfalgo_config() {}
};

void pfalgo_mu_ref(const pfalgo_config &cfg,
                   const TkObj track[/*cfg.nTRACK*/],
                   const MuObj mu[/*cfg.nMU*/],
                   bool isMu[/*cfg.nTRACK*/],
                   PFChargedObj outmu[/*cfg.nMU*/],
                   bool debug);

//=== begin implementation part

template <typename CO_t>
int best_match_with_pt_ref(int nCAL, int dR2MAX, const CO_t calo[/*nCAL*/], const TkObj &track) {
  pt_t caloPtMin = track.hwPt - 2 * (track.hwPtErr);
  if (caloPtMin < 0)
    caloPtMin = 0;
  int dptscale = (dR2MAX << 8) / std::max<int>(1, sqr(track.hwPtErr));
  int drmin = 0, ibest = -1;
  for (int ic = 0; ic < nCAL; ++ic) {
    if (calo[ic].hwPt <= caloPtMin)
      continue;
    int dr = dr2_int(track.hwEta, track.hwPhi, calo[ic].hwEta, calo[ic].hwPhi);
    if (dr >= dR2MAX)
      continue;
    dr += ((sqr(std::max<int>(track.hwPt - calo[ic].hwPt, 0)) * dptscale) >> 8);
    if (ibest == -1 || dr < drmin) {
      drmin = dr;
      ibest = ic;
    }
  }
  return ibest;
}

template <typename T, typename TV>
void ptsort_ref(int nIn, int nOut, const TV &in /*[nIn]*/, T out[/*nOut*/]) {
  for (int iout = 0; iout < nOut; ++iout) {
    out[iout].hwPt = 0;
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
