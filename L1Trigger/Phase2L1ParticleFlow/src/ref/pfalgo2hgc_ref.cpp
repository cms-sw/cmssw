#include "pfalgo2hgc_ref.h"

#ifndef CMSSW_GIT_HASH
#include "../DiscretePFInputs.h"
#else
#include "../../interface/DiscretePFInputs.h"
#endif

#include "../utils/Firmware2DiscretePF.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <memory>

void pfalgo2hgc_ref(const pfalgo_config &cfg,
                    const HadCaloObj calo[/*cfg.nCALO*/],
                    const TkObj track[/*cfg.nTRACK*/],
                    const MuObj mu[/*cfg.nMU*/],
                    PFChargedObj outch[/*cfg.nTRACK*/],
                    PFNeutralObj outne[/*cfg.nSELCALO*/],
                    PFChargedObj outmu[/*cfg.nMU*/],
                    bool debug) {
  if (debug) {
#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    for (unsigned int i = 0; i < cfg.nTRACK; ++i) {
      if (track[i].hwPt == 0)
        continue;
      l1tpf_impl::PropagatedTrack tk;
      fw2dpf::convert(track[i], tk);
      printf(
          "FW  \t track %3d: pt %8d [ %7.2f ]  calo eta %+7d [ %+5.2f ]  calo phi %+7d [ %+5.2f ]  calo ptErr %6d [ "
          "%7.2f ]   tight %d\n",
          i,
          tk.hwPt,
          tk.floatPt(),
          tk.hwEta,
          tk.floatEta(),
          tk.hwPhi,
          tk.floatPhi(),
          tk.hwCaloPtErr,
          tk.floatCaloPtErr(),
          int(track[i].hwTightQuality));
    }
    for (unsigned int i = 0; i < cfg.nCALO; ++i) {
      if (calo[i].hwPt == 0)
        continue;
      l1tpf_impl::CaloCluster c;
      fw2dpf::convert(calo[i], c);
      printf(
          "FW  \t calo  %3d: pt %8d [ %7.2f ]  calo eta %+7d [ %+5.2f ]  calo phi %+7d [ %+5.2f ]  calo emPt %7d [ "
          "%7.2f ]   isEM %d \n",
          i,
          c.hwPt,
          c.floatPt(),
          c.hwEta,
          c.floatEta(),
          c.hwPhi,
          c.floatPhi(),
          c.hwEmPt,
          c.floatEmPt(),
          c.isEM);
    }
    for (unsigned int i = 0; i < cfg.nMU; ++i) {
      if (mu[i].hwPt == 0)
        continue;
      l1tpf_impl::Muon muon;
      fw2dpf::convert(mu[i], muon);
      printf("FW  \t muon  %3d: pt %8d [ %7.2f ]  muon eta %+7d [ %+5.2f ]  muon phi %+7d [ %+5.2f ]   \n",
             i,
             muon.hwPt,
             muon.floatPt(),
             muon.hwEta,
             muon.floatEta(),
             muon.hwPhi,
             muon.floatPhi());
    }
#endif
  }

  // constants
  const pt_t TKPT_MAX_LOOSE = cfg.tk_MAXINVPT_LOOSE;
  const pt_t TKPT_MAX_TIGHT = cfg.tk_MAXINVPT_TIGHT;
  const int DR2MAX = cfg.dR2MAX_TK_CALO;

  ////////////////////////////////////////////////////
  // TK-MU Linking
  std::unique_ptr<bool[]> isMu(new bool[cfg.nTRACK]);
  pfalgo_mu_ref(cfg, track, mu, &isMu[0], outmu, debug);

  ////////////////////////////////////////////////////
  // TK-HAD Linking

  // initialize sum track pt
  std::vector<pt_t> calo_sumtk(cfg.nCALO), calo_subpt(cfg.nCALO);
  std::vector<int> calo_sumtkErr2(cfg.nCALO);
  for (unsigned int ic = 0; ic < cfg.nCALO; ++ic) {
    calo_sumtk[ic] = 0;
    calo_sumtkErr2[ic] = 0;
  }

  // initialize good track bit
  std::unique_ptr<bool[]> track_good(new bool[cfg.nTRACK]);
  std::unique_ptr<bool[]> isEle(new bool[cfg.nTRACK]);
  for (unsigned int it = 0; it < cfg.nTRACK; ++it) {
    track_good[it] = (track[it].hwPt < (track[it].hwTightQuality ? TKPT_MAX_TIGHT : TKPT_MAX_LOOSE) || isMu[it]);
    isEle[it] = false;
  }

  // initialize output
  for (unsigned int ipf = 0; ipf < cfg.nTRACK; ++ipf)
    clear(outch[ipf]);
  for (unsigned int ipf = 0; ipf < cfg.nSELCALO; ++ipf)
    clear(outne[ipf]);

  // for each track, find the closest calo
  for (unsigned int it = 0; it < cfg.nTRACK; ++it) {
    if (track[it].hwPt > 0 && !isMu[it]) {
      int ibest = best_match_with_pt_ref<HadCaloObj>(cfg.nCALO, DR2MAX, calo, track[it]);
      if (ibest != -1) {
        if (debug)
          printf("FW  \t track  %3d pt %7d matched to calo' %3d pt %7d\n",
                 it,
                 int(track[it].hwPt),
                 ibest,
                 int(calo[ibest].hwPt));
        track_good[it] = true;
        isEle[it] = calo[ibest].hwIsEM;
        calo_sumtk[ibest] += track[it].hwPt;
        calo_sumtkErr2[ibest] += sqr(track[it].hwPtErr);
      }
    }
  }

  for (unsigned int ic = 0; ic < cfg.nCALO; ++ic) {
    if (calo_sumtk[ic] > 0) {
      pt_t ptdiff = calo[ic].hwPt - calo_sumtk[ic];
      int sigmamult =
          calo_sumtkErr2[ic];  //  + (calo_sumtkErr2[ic] >> 1)); // this multiplies by 1.5 = sqrt(1.5)^2 ~ (1.2)^2
      if (debug && (calo[ic].hwPt > 0)) {
#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
        l1tpf_impl::CaloCluster floatcalo;
        fw2dpf::convert(calo[ic], floatcalo);
        printf(
            "FW  \t calo'  %3d pt %7d [ %7.2f ] eta %+7d [ %+5.2f ] has a sum track pt %7d, difference %7d +- %.2f \n",
            ic,
            int(calo[ic].hwPt),
            floatcalo.floatPt(),
            int(calo[ic].hwEta),
            floatcalo.floatEta(),
            int(calo_sumtk[ic]),
            int(ptdiff),
            std::sqrt(float(int(calo_sumtkErr2[ic]))));
#endif
      }
      if (ptdiff > 0 && ptdiff * ptdiff > sigmamult) {
        calo_subpt[ic] = ptdiff;
      } else {
        calo_subpt[ic] = 0;
      }
    } else {
      calo_subpt[ic] = calo[ic].hwPt;
    }
    if (debug && (calo[ic].hwPt > 0))
      printf("FW  \t calo'  %3d pt %7d ---> %7d \n", ic, int(calo[ic].hwPt), int(calo_subpt[ic]));
  }

  // copy out charged hadrons
  for (unsigned int it = 0; it < cfg.nTRACK; ++it) {
    if (track_good[it]) {
      assert(!(isEle[it] && isMu[it]));
      outch[it].hwPt = track[it].hwPt;
      outch[it].hwEta = track[it].hwEta;
      outch[it].hwPhi = track[it].hwPhi;
      outch[it].hwZ0 = track[it].hwZ0;
      outch[it].hwId = isEle[it] ? PID_Electron : (isMu[it] ? PID_Muon : PID_Charged);
    }
  }

  // copy out neutral hadrons with sorting and cropping
  std::vector<PFNeutralObj> outne_all(cfg.nCALO);
  for (unsigned int ipf = 0; ipf < cfg.nCALO; ++ipf)
    clear(outne_all[ipf]);
  for (unsigned int ic = 0; ic < cfg.nCALO; ++ic) {
    if (calo_subpt[ic] > 0) {
      outne_all[ic].hwPt = calo_subpt[ic];
      outne_all[ic].hwEta = calo[ic].hwEta;
      outne_all[ic].hwPhi = calo[ic].hwPhi;
      outne_all[ic].hwId = calo[ic].hwIsEM ? PID_Photon : PID_Neutral;
    }
  }

  ptsort_ref(cfg.nCALO, cfg.nSELCALO, outne_all, outne);
}
