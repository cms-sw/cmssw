#include "pfalgo3_ref.h"

#ifndef CMSSW_GIT_HASH
#include "../DiscretePFInputs.h"
#else
#include "../../interface/DiscretePFInputs.h"
#endif

#include "../utils/Firmware2DiscretePF.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <memory>

template <bool doPtMin, typename CO_t>
int tk_best_match_ref(unsigned int nCAL, unsigned int dR2MAX, const CO_t calo[/*nCAL*/], const TkObj &track) {
  pt_t caloPtMin = track.hwPt - 2 * (track.hwPtErr);
  if (caloPtMin < 0)
    caloPtMin = 0;
  int drmin = dR2MAX, ibest = -1;
  for (unsigned int ic = 0; ic < nCAL; ++ic) {
    if (calo[ic].hwPt <= 0)
      continue;
    if (doPtMin && calo[ic].hwPt <= caloPtMin)
      continue;
    int dr = dr2_int(track.hwEta, track.hwPhi, calo[ic].hwEta, calo[ic].hwPhi);
    if (dr < drmin) {
      drmin = dr;
      ibest = ic;
    }
  }
  return ibest;
}
int em_best_match_ref(unsigned int nCAL, unsigned int dR2MAX, const HadCaloObj calo[/*nCAL*/], const EmCaloObj &em) {
  pt_t emPtMin = em.hwPt >> 1;
  int drmin = dR2MAX, ibest = -1;
  for (unsigned int ic = 0; ic < nCAL; ++ic) {
    if (calo[ic].hwEmPt <= emPtMin)
      continue;
    int dr = dr2_int(em.hwEta, em.hwPhi, calo[ic].hwEta, calo[ic].hwPhi);
    if (dr < drmin) {
      drmin = dr;
      ibest = ic;
    }
  }
  return ibest;
}

void pfalgo3_em_ref(const pfalgo3_config &cfg,
                    const EmCaloObj emcalo[/*cfg.nEMCALO*/],
                    const HadCaloObj hadcalo[/*cfg.nCALO*/],
                    const TkObj track[/*cfg.nTRACK*/],
                    const bool isMu[/*cfg.nTRACK*/],
                    bool isEle[/*cfg.nTRACK*/],
                    PFNeutralObj outpho[/*cfg.nPHOTON*/],
                    HadCaloObj hadcalo_out[/*cfg.nCALO*/],
                    bool debug) {
  // constants
  const int DR2MAX_TE = cfg.dR2MAX_TK_EM;
  const int DR2MAX_EH = cfg.dR2MAX_EM_CALO;

  // initialize sum track pt
  std::vector<pt_t> calo_sumtk(cfg.nEMCALO);
  for (unsigned int ic = 0; ic < cfg.nEMCALO; ++ic) {
    calo_sumtk[ic] = 0;
  }
  std::vector<int> tk2em(cfg.nTRACK);
  std::vector<bool> isEM(cfg.nEMCALO);
  // for each track, find the closest calo
  for (unsigned int it = 0; it < cfg.nTRACK; ++it) {
    if (track[it].hwPt > 0 && !isMu[it]) {
      tk2em[it] = tk_best_match_ref<false, EmCaloObj>(cfg.nEMCALO, DR2MAX_TE, emcalo, track[it]);
      if (tk2em[it] != -1) {
        if (debug)
          printf("FW  \t track  %3d pt %7d matched to em calo %3d pt %7d\n",
                 it,
                 int(track[it].hwPt),
                 tk2em[it],
                 int(emcalo[tk2em[it]].hwPt));
        calo_sumtk[tk2em[it]] += track[it].hwPt;
      }
    } else {
      tk2em[it] = -1;
    }
  }

  if (debug) {
    for (unsigned int ic = 0; ic < cfg.nEMCALO; ++ic) {
      if (emcalo[ic].hwPt > 0)
        printf("FW  \t emcalo %3d pt %7d has sumtk %7d\n", ic, int(emcalo[ic].hwPt), int(calo_sumtk[ic]));
    }
  }

  for (unsigned int ic = 0; ic < cfg.nEMCALO; ++ic) {
    pt_t photonPt;
    if (calo_sumtk[ic] > 0) {
      pt_t ptdiff = emcalo[ic].hwPt - calo_sumtk[ic];
      int sigma2 = sqr(emcalo[ic].hwPtErr);
      int sigma2Lo = 4 * sigma2,
          sigma2Hi = sigma2;  // + (sigma2>>1); // cut at 1 sigma instead of old cut at sqrt(1.5) sigma's
      int ptdiff2 = ptdiff * ptdiff;
      if ((ptdiff >= 0 && ptdiff2 <= sigma2Hi) || (ptdiff < 0 && ptdiff2 < sigma2Lo)) {
        // electron
        photonPt = 0;
        isEM[ic] = true;
        if (debug)
          printf("FW  \t emcalo %3d pt %7d ptdiff %7d [match window: -%.2f / +%.2f] flagged as electron\n",
                 ic,
                 int(emcalo[ic].hwPt),
                 int(ptdiff),
                 std::sqrt(float(sigma2Lo)),
                 std::sqrt(float(sigma2Hi)));
      } else if (ptdiff > 0) {
        // electron + photon
        photonPt = ptdiff;
        isEM[ic] = true;
        if (debug)
          printf(
              "FW  \t emcalo %3d pt %7d ptdiff %7d [match window: -%.2f / +%.2f] flagged as electron + photon of pt "
              "%7d\n",
              ic,
              int(emcalo[ic].hwPt),
              int(ptdiff),
              std::sqrt(float(sigma2Lo)),
              std::sqrt(float(sigma2Hi)),
              int(photonPt));
      } else {
        // pion
        photonPt = 0;
        isEM[ic] = false;
        if (debug)
          printf("FW  \t emcalo %3d pt %7d ptdiff %7d [match window: -%.2f / +%.2f] flagged as pion\n",
                 ic,
                 int(emcalo[ic].hwPt),
                 int(ptdiff),
                 std::sqrt(float(sigma2Lo)),
                 std::sqrt(float(sigma2Hi)));
      }
    } else {
      // photon
      isEM[ic] = true;
      photonPt = emcalo[ic].hwPt;
      if (debug && emcalo[ic].hwPt > 0)
        printf("FW  \t emcalo %3d pt %7d flagged as photon\n", ic, int(emcalo[ic].hwPt));
    }
    outpho[ic].hwPt = photonPt;
    outpho[ic].hwEta = photonPt ? emcalo[ic].hwEta : etaphi_t(0);
    outpho[ic].hwPhi = photonPt ? emcalo[ic].hwPhi : etaphi_t(0);
    outpho[ic].hwId = photonPt ? PID_Photon : particleid_t(0);
  }

  for (unsigned int it = 0; it < cfg.nTRACK; ++it) {
    isEle[it] = (tk2em[it] != -1) && isEM[tk2em[it]];
    if (debug && isEle[it])
      printf("FW  \t track  %3d pt %7d flagged as electron.\n", it, int(track[it].hwPt));
  }

  std::vector<int> em2calo(cfg.nEMCALO);
  for (unsigned int ic = 0; ic < cfg.nEMCALO; ++ic) {
    em2calo[ic] = em_best_match_ref(cfg.nCALO, DR2MAX_EH, hadcalo, emcalo[ic]);
    if (debug && (emcalo[ic].hwPt > 0)) {
      printf("FW  \t emcalo %3d pt %7d isEM %d matched to hadcalo %7d pt %7d emPt %7d isEM %d\n",
             ic,
             int(emcalo[ic].hwPt),
             int(isEM[ic]),
             em2calo[ic],
             (em2calo[ic] >= 0 ? int(hadcalo[em2calo[ic]].hwPt) : -1),
             (em2calo[ic] >= 0 ? int(hadcalo[em2calo[ic]].hwEmPt) : -1),
             (em2calo[ic] >= 0 ? int(hadcalo[em2calo[ic]].hwIsEM) : 0));
    }
  }

  for (unsigned int ih = 0; ih < cfg.nCALO; ++ih) {
    hadcalo_out[ih] = hadcalo[ih];
    pt_t sub = 0;
    bool keep = false;
    for (unsigned int ic = 0; ic < cfg.nEMCALO; ++ic) {
      if (em2calo[ic] == int(ih)) {
        if (isEM[ic])
          sub += emcalo[ic].hwPt;
        else
          keep = true;
      }
    }
    pt_t emdiff = hadcalo[ih].hwEmPt - sub;
    pt_t alldiff = hadcalo[ih].hwPt - sub;
    if (debug && (hadcalo[ih].hwPt > 0)) {
      printf("FW  \t calo   %3d pt %7d has a subtracted pt of %7d, empt %7d -> %7d   isem %d mustkeep %d \n",
             ih,
             int(hadcalo[ih].hwPt),
             int(alldiff),
             int(hadcalo[ih].hwEmPt),
             int(emdiff),
             int(hadcalo[ih].hwIsEM),
             keep);
    }
    if (alldiff <= (hadcalo[ih].hwPt >> 4)) {
      hadcalo_out[ih].hwPt = 0;    // kill
      hadcalo_out[ih].hwEmPt = 0;  // kill
      if (debug && (hadcalo[ih].hwPt > 0))
        printf("FW  \t calo   %3d pt %7d --> discarded (zero pt)\n", ih, int(hadcalo[ih].hwPt));
    } else if ((hadcalo[ih].hwIsEM && emdiff <= (hadcalo[ih].hwEmPt >> 3)) && !keep) {
      hadcalo_out[ih].hwPt = 0;    // kill
      hadcalo_out[ih].hwEmPt = 0;  // kill
      if (debug && (hadcalo[ih].hwPt > 0))
        printf("FW  \t calo   %3d pt %7d --> discarded (zero em)\n", ih, int(hadcalo[ih].hwPt));
    } else {
      hadcalo_out[ih].hwPt = alldiff;
      hadcalo_out[ih].hwEmPt = (emdiff > 0 ? emdiff : pt_t(0));
    }
  }
}

void pfalgo3_ref(const pfalgo3_config &cfg,
                 const EmCaloObj emcalo[/*cfg.nEMCALO*/],
                 const HadCaloObj hadcalo[/*cfg.nCALO*/],
                 const TkObj track[/*cfg.nTRACK*/],
                 const MuObj mu[/*cfg.nMU*/],
                 PFChargedObj outch[/*cfg.nTRACK*/],
                 PFNeutralObj outpho[/*cfg.nPHOTON*/],
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
    for (unsigned int i = 0; i < cfg.nEMCALO; ++i) {
      if (emcalo[i].hwPt == 0)
        continue;
      l1tpf_impl::CaloCluster em;
      fw2dpf::convert(emcalo[i], em);
      printf(
          "FW  \t EM    %3d: pt %8d [ %7.2f ]  calo eta %+7d [ %+5.2f ]  calo phi %+7d [ %+5.2f ]  calo ptErr %6d [ "
          "%7.2f ] \n",
          i,
          em.hwPt,
          em.floatPt(),
          em.hwEta,
          em.floatEta(),
          em.hwPhi,
          em.floatPhi(),
          em.hwPtErr,
          em.floatPtErr());
    }
    for (unsigned int i = 0; i < cfg.nCALO; ++i) {
      if (hadcalo[i].hwPt == 0)
        continue;
      l1tpf_impl::CaloCluster calo;
      fw2dpf::convert(hadcalo[i], calo);
      printf(
          "FW  \t calo  %3d: pt %8d [ %7.2f ]  calo eta %+7d [ %+5.2f ]  calo phi %+7d [ %+5.2f ]  calo emPt %7d [ "
          "%7.2f ]   isEM %d \n",
          i,
          calo.hwPt,
          calo.floatPt(),
          calo.hwEta,
          calo.floatEta(),
          calo.hwPhi,
          calo.floatPhi(),
          calo.hwEmPt,
          calo.floatEmPt(),
          calo.isEM);
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
  // // we can't use std::vector here because it's specialized
  std::unique_ptr<bool[]> isMu(new bool[cfg.nTRACK]);
  pfalgo_mu_ref(cfg, track, mu, &isMu[0], outmu, debug);

  ////////////////////////////////////////////////////
  // TK-EM Linking
  std::unique_ptr<bool[]> isEle(new bool[cfg.nTRACK]);
  std::vector<HadCaloObj> hadcalo_subem(cfg.nCALO);
  pfalgo3_em_ref(cfg, emcalo, hadcalo, track, &isMu[0], &isEle[0], outpho, &hadcalo_subem[0], debug);

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
  for (unsigned int it = 0; it < cfg.nTRACK; ++it) {
    track_good[it] =
        (track[it].hwPt < (track[it].hwTightQuality ? TKPT_MAX_TIGHT : TKPT_MAX_LOOSE) || isEle[it] || isMu[it]);
  }

  // initialize output
  for (unsigned int ipf = 0; ipf < cfg.nTRACK; ++ipf) {
    clear(outch[ipf]);
  }
  for (unsigned int ipf = 0; ipf < cfg.nSELCALO; ++ipf) {
    clear(outne[ipf]);
  }

  // for each track, find the closest calo
  for (unsigned int it = 0; it < cfg.nTRACK; ++it) {
    if (track[it].hwPt > 0 && !isEle[it] && !isMu[it]) {
      int ibest = best_match_with_pt_ref<HadCaloObj>(cfg.nCALO, DR2MAX, &hadcalo_subem[0], track[it]);
      //int  ibest = tk_best_match_ref<true,HadCaloObj>(cfg.nCALO, DR2MAX, &hadcalo_subem[0], track[it]);
      if (ibest != -1) {
        if (debug)
          printf("FW  \t track  %3d pt %7d matched to calo %3d pt %7d\n",
                 it,
                 int(track[it].hwPt),
                 ibest,
                 int(hadcalo_subem[ibest].hwPt));
        track_good[it] = true;
        calo_sumtk[ibest] += track[it].hwPt;
        calo_sumtkErr2[ibest] += sqr(track[it].hwPtErr);
      }
    }
  }

  for (unsigned int ic = 0; ic < cfg.nCALO; ++ic) {
    if (calo_sumtk[ic] > 0) {
      pt_t ptdiff = hadcalo_subem[ic].hwPt - calo_sumtk[ic];
      int sigmamult = calo_sumtkErr2
          [ic];  // before we did (calo_sumtkErr2[ic] + (calo_sumtkErr2[ic] >> 1)); to multiply by 1.5 = sqrt(1.5)^2 ~ (1.2)^2
      if (debug && (hadcalo_subem[ic].hwPt > 0)) {
#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
        l1tpf_impl::CaloCluster floatcalo;
        fw2dpf::convert(hadcalo_subem[ic], floatcalo);
        printf(
            "FW  \t calo  %3d pt %7d [ %7.2f ] eta %+7d [ %+5.2f ] has a sum track pt %7d, difference %7d +- %.2f \n",
            ic,
            int(hadcalo_subem[ic].hwPt),
            floatcalo.floatPt(),
            int(hadcalo_subem[ic].hwEta),
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
      calo_subpt[ic] = hadcalo_subem[ic].hwPt;
    }
    if (debug && (hadcalo_subem[ic].hwPt > 0))
      printf("FW  \t calo  %3d pt %7d ---> %7d \n", ic, int(hadcalo_subem[ic].hwPt), int(calo_subpt[ic]));
  }

  // copy out charged hadrons
  for (unsigned int it = 0; it < cfg.nTRACK; ++it) {
    if (track_good[it]) {
      outch[it].hwPt = track[it].hwPt;
      outch[it].hwEta = track[it].hwEta;
      outch[it].hwPhi = track[it].hwPhi;
      outch[it].hwZ0 = track[it].hwZ0;
      outch[it].hwId = isEle[it] ? PID_Electron : (isMu[it] ? PID_Muon : PID_Charged);
    }
  }

  // copy out neutral hadrons
  std::vector<PFNeutralObj> outne_all(cfg.nCALO);
  for (unsigned int ipf = 0; ipf < cfg.nCALO; ++ipf) {
    clear(outne_all[ipf]);
  }
  for (unsigned int ic = 0; ic < cfg.nCALO; ++ic) {
    if (calo_subpt[ic] > 0) {
      outne_all[ic].hwPt = calo_subpt[ic];
      outne_all[ic].hwEta = hadcalo_subem[ic].hwEta;
      outne_all[ic].hwPhi = hadcalo_subem[ic].hwPhi;
      outne_all[ic].hwId = PID_Neutral;
    }
  }

  ptsort_ref(cfg.nCALO, cfg.nSELCALO, outne_all, outne);

  if (debug) {
#ifdef L1Trigger_Phase2L1ParticleFlow_DiscretePFInputs_MORE
    std::vector<l1tpf_impl::PFParticle> tmp;
    for (unsigned int i = 0; i < cfg.nTRACK; ++i) {
      if (outch[i].hwPt == 0)
        continue;
      fw2dpf::convert(outch[i], track[i], tmp);
      auto &pf = tmp.back();
      printf("FW  \t outch %3d: pt %8d [ %7.2f ]  calo eta %+7d [ %+5.2f ]  calo phi %+7d [ %+5.2f ]  pid %d\n",
             i,
             pf.hwPt,
             pf.floatPt(),
             pf.hwEta,
             pf.floatEta(),
             pf.hwPhi,
             pf.floatPhi(),
             pf.hwId);
    }
    for (unsigned int i = 0; i < cfg.nPHOTON; ++i) {
      if (outpho[i].hwPt == 0)
        continue;
      fw2dpf::convert(outpho[i], tmp);
      auto &pf = tmp.back();
      printf("FW  \t outph %3d: pt %8d [ %7.2f ]  calo eta %+7d [ %+5.2f ]  calo phi %+7d [ %+5.2f ]  pid %d\n",
             i,
             pf.hwPt,
             pf.floatPt(),
             pf.hwEta,
             pf.floatEta(),
             pf.hwPhi,
             pf.floatPhi(),
             pf.hwId);
    }
    for (unsigned int i = 0; i < cfg.nSELCALO; ++i) {
      if (outne[i].hwPt == 0)
        continue;
      fw2dpf::convert(outne[i], tmp);
      auto &pf = tmp.back();
      printf("FW  \t outne %3d: pt %8d [ %7.2f ]  calo eta %+7d [ %+5.2f ]  calo phi %+7d [ %+5.2f ]  pid %d\n",
             i,
             pf.hwPt,
             pf.floatPt(),
             pf.hwEta,
             pf.floatEta(),
             pf.hwPhi,
             pf.floatPhi(),
             pf.hwId);
    }
#endif
  }
}

void pfalgo3_merge_neutrals_ref(const pfalgo3_config &cfg,
                                const PFNeutralObj pho[/*cfg.nPHOTON*/],
                                const PFNeutralObj ne[/*cfg.nSELCALO*/],
                                PFNeutralObj allne[/*cfg.nALLNEUTRALS*/]) {
  int j = 0;
  for (unsigned int i = 0; i < cfg.nPHOTON; ++i, ++j)
    allne[j] = pho[i];
  for (unsigned int i = 0; i < cfg.nSELCALO; ++i, ++j)
    allne[j] = ne[i];
}
