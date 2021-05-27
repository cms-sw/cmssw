#include "pfalgo3_ref.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <memory>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::PFAlgo3Emulator::PFAlgo3Emulator(const edm::ParameterSet& iConfig)
    : PFAlgoEmulatorBase(iConfig.getParameter<uint32_t>("nTrack"),
                         iConfig.getParameter<uint32_t>("nCalo"),
                         iConfig.getParameter<uint32_t>("nMu"),
                         iConfig.getParameter<uint32_t>("nSelCalo"),
                         l1ct::Scales::makeDR2FromFloatDR(iConfig.getParameter<double>("trackMuDR")),
                         l1ct::Scales::makeDR2FromFloatDR(iConfig.getParameter<double>("trackCaloDR")),
                         l1ct::Scales::makePtFromFloat(iConfig.getParameter<double>("maxInvisiblePt")),
                         l1ct::Scales::makePtFromFloat(iConfig.getParameter<double>("tightTrackMaxInvisiblePt"))),
      nEMCALO_(iConfig.getParameter<uint32_t>("nEmCalo")),
      nPHOTON_(iConfig.getParameter<uint32_t>("nPhoton")),
      nALLNEUTRAL_(iConfig.getParameter<uint32_t>("nAllNeutral")),
      dR2MAX_TK_EM_(l1ct::Scales::makeDR2FromFloatDR(iConfig.getParameter<double>("trackEmDR"))),
      dR2MAX_EM_CALO_(l1ct::Scales::makeDR2FromFloatDR(iConfig.getParameter<double>("emCaloDR"))) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  loadPtErrBins(iConfig);
}
#endif

void l1ct::PFAlgo3Emulator::toFirmware(const PFInputRegion& in,
                                       PFRegion& region,
                                       HadCaloObj calo[/*nCALO*/],
                                       EmCaloObj emcalo[/*nEMCALO*/],
                                       TkObj track[/*nTRACK*/],
                                       MuObj mu[/*nMU*/]) const {
  region = in.region;
  l1ct::toFirmware(in.track, nTRACK_, track);
  l1ct::toFirmware(in.emcalo, nEMCALO_, emcalo);
  l1ct::toFirmware(in.hadcalo, nCALO_, calo);
  l1ct::toFirmware(in.muon, nMU_, mu);
}

void l1ct::PFAlgo3Emulator::toFirmware(const OutputRegion& out,
                                       PFChargedObj outch[/*nTRACK*/],
                                       PFNeutralObj outpho[/*nPHOTON*/],
                                       PFNeutralObj outne[/*nSELCALO*/],
                                       PFChargedObj outmu[/*nMU*/]) const {
  l1ct::toFirmware(out.pfcharged, nTRACK_, outch);
  l1ct::toFirmware(out.pfphoton, nPHOTON_, outpho);
  l1ct::toFirmware(out.pfneutral, nSELCALO_, outne);
  l1ct::toFirmware(out.pfmuon, nMU_, outmu);
}

int l1ct::PFAlgo3Emulator::tk_best_match_ref(unsigned int dR2MAX,
                                             const std::vector<l1ct::EmCaloObjEmu>& calo,
                                             const l1ct::TkObjEmu& track) const {
  int drmin = dR2MAX, ibest = -1;
  for (unsigned int ic = 0, nCAL = calo.size(); ic < nCAL; ++ic) {
    if (calo[ic].hwPt <= 0)
      continue;
    int dr = dr2_int(track.hwEta, track.hwPhi, calo[ic].hwEta, calo[ic].hwPhi);
    if (dr < drmin) {
      drmin = dr;
      ibest = ic;
    }
  }
  return ibest;
}
int l1ct::PFAlgo3Emulator::em_best_match_ref(unsigned int dR2MAX,
                                             const std::vector<l1ct::HadCaloObjEmu>& calo,
                                             const l1ct::EmCaloObjEmu& em) const {
  pt_t emPtMin = em.hwPt >> 1;
  int drmin = dR2MAX, ibest = -1;
  for (unsigned int ic = 0, nCAL = calo.size(); ic < nCAL; ++ic) {
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

void l1ct::PFAlgo3Emulator::pfalgo3_em_ref(const PFInputRegion& in,
                                           const std::vector<int>& iMu /*[nTRACK]*/,
                                           std::vector<int>& iEle /*[nTRACK]*/,
                                           OutputRegion& out,
                                           std::vector<HadCaloObjEmu>& hadcalo_out /*[nCALO]*/) const {
  // constants
  unsigned int nTRACK = std::min<unsigned>(nTRACK_, in.track.size());
  unsigned int nEMCALO = std::min<unsigned>(nEMCALO_, in.emcalo.size());
  unsigned int nPHOTON = std::min<unsigned>(nPHOTON_, in.emcalo.size());
  unsigned int nCALO = std::min<unsigned>(nCALO_, in.hadcalo.size());

  // initialize sum track pt
  std::vector<pt_t> calo_sumtk(nEMCALO);
  for (unsigned int ic = 0; ic < nEMCALO; ++ic) {
    calo_sumtk[ic] = 0;
  }
  std::vector<int> tk2em(nTRACK);
  std::vector<bool> isEM(nEMCALO);
  // for each track, find the closest calo
  for (unsigned int it = 0; it < nTRACK; ++it) {
    if (in.track[it].hwPt > 0 && in.track[it].isPFLoose() && iMu[it] == -1) {
      tk2em[it] = tk_best_match_ref(dR2MAX_TK_EM_, in.emcalo, in.track[it]);
      if (tk2em[it] != -1) {
        if (debug_)
          printf(
              "FW  \t track  %3d pt %8.2f matched to em calo %3d pt %8.2f (int deltaR2 %d)\n",
              it,
              in.track[it].floatPt(),
              tk2em[it],
              in.emcalo[tk2em[it]].floatPt(),
              dr2_int(in.track[it].hwEta, in.track[it].hwPhi, in.emcalo[tk2em[it]].hwEta, in.emcalo[tk2em[it]].hwPhi));
        calo_sumtk[tk2em[it]] += in.track[it].hwPt;
      }
    } else {
      tk2em[it] = -1;
    }
  }

  if (debug_) {
    for (unsigned int ic = 0; ic < nEMCALO; ++ic) {
      if (in.emcalo[ic].hwPt > 0)
        printf("FW  \t emcalo %3d pt %8.2f has sumtk %8.2f\n",
               ic,
               in.emcalo[ic].floatPt(),
               Scales::floatPt(calo_sumtk[ic]));
    }
  }

  assert(nEMCALO == nPHOTON);  // code doesn't work otherwise
  out.pfphoton.resize(nPHOTON);
  for (unsigned int ic = 0; ic < nEMCALO; ++ic) {
    pt_t photonPt;
    if (calo_sumtk[ic] > 0) {
      dpt_t ptdiff = dpt_t(in.emcalo[ic].hwPt) - dpt_t(calo_sumtk[ic]);
      pt2_t sigma2 = in.emcalo[ic].hwPtErr * in.emcalo[ic].hwPtErr;
      pt2_t sigma2Lo = 4 * sigma2,
            sigma2Hi = sigma2;  // + (sigma2>>1); // cut at 1 sigma instead of old cut at sqrt(1.5) sigma's
      pt2_t ptdiff2 = ptdiff * ptdiff;
      if ((ptdiff >= 0 && ptdiff2 <= sigma2Hi) || (ptdiff < 0 && ptdiff2 < sigma2Lo)) {
        // electron
        photonPt = 0;
        isEM[ic] = true;
        if (debug_)
          printf("FW  \t emcalo %3d pt %8.2f ptdiff %8.2f [match window: -%.2f / +%.2f] flagged as electron\n",
                 ic,
                 in.emcalo[ic].floatPt(),
                 Scales::floatPt(ptdiff),
                 std::sqrt(Scales::floatPt(sigma2Lo)),
                 std::sqrt(float(sigma2Hi)));
      } else if (ptdiff > 0) {
        // electron + photon
        photonPt = ptdiff;
        isEM[ic] = true;
        if (debug_)
          printf(
              "FW  \t emcalo %3d pt %8.2f ptdiff %8.2f [match window: -%.2f / +%.2f] flagged as electron + photon of "
              "pt %8.2f\n",
              ic,
              in.emcalo[ic].floatPt(),
              Scales::floatPt(ptdiff),
              std::sqrt(Scales::floatPt(sigma2Lo)),
              std::sqrt(float(sigma2Hi)),
              Scales::floatPt(photonPt));
      } else {
        // pion
        photonPt = 0;
        isEM[ic] = false;
        if (debug_)
          printf("FW  \t emcalo %3d pt %8.2f ptdiff %8.2f [match window: -%.2f / +%.2f] flagged as pion\n",
                 ic,
                 in.emcalo[ic].floatPt(),
                 Scales::floatPt(ptdiff),
                 std::sqrt(Scales::floatPt(sigma2Lo)),
                 std::sqrt(Scales::floatPt(sigma2Hi)));
      }
    } else {
      // photon
      isEM[ic] = true;
      photonPt = in.emcalo[ic].hwPt;
      if (debug_ && in.emcalo[ic].hwPt > 0)
        printf("FW  \t emcalo %3d pt %8.2f flagged as photon\n", ic, in.emcalo[ic].floatPt());
    }
    if (photonPt) {
      fillPFCand(in.emcalo[ic], out.pfphoton[ic]);
      out.pfphoton[ic].hwPt = photonPt;
      out.pfphoton[ic].hwEmPt = photonPt;
    } else {
      out.pfphoton[ic].clear();
    }
  }

  iEle.resize(nTRACK);
  for (unsigned int it = 0; it < nTRACK; ++it) {
    iEle[it] = ((tk2em[it] != -1) && isEM[tk2em[it]]) ? tk2em[it] : -1;
    if (debug_ && (iEle[it] != -1))
      printf("FW  \t track  %3d pt %8.2f flagged as electron (emcluster %d).\n", it, in.track[it].floatPt(), iEle[it]);
  }

  std::vector<int> em2calo(nEMCALO);
  for (unsigned int ic = 0; ic < nEMCALO; ++ic) {
    em2calo[ic] = em_best_match_ref(dR2MAX_EM_CALO_, in.hadcalo, in.emcalo[ic]);
    if (debug_ && (in.emcalo[ic].hwPt > 0)) {
      printf("FW  \t emcalo %3d pt %8.2f isEM %d matched to hadcalo %3d pt %8.2f emPt %8.2f isEM %d\n",
             ic,
             in.emcalo[ic].floatPt(),
             int(isEM[ic]),
             em2calo[ic],
             (em2calo[ic] >= 0 ? in.hadcalo[em2calo[ic]].floatPt() : -1),
             (em2calo[ic] >= 0 ? in.hadcalo[em2calo[ic]].floatEmPt() : -1),
             (em2calo[ic] >= 0 ? int(in.hadcalo[em2calo[ic]].hwIsEM) : 0));
    }
  }

  hadcalo_out.resize(nCALO);
  for (unsigned int ih = 0; ih < nCALO; ++ih) {
    hadcalo_out[ih] = in.hadcalo[ih];
    dpt_t sub = 0;
    bool keep = false;
    for (unsigned int ic = 0; ic < nEMCALO; ++ic) {
      if (em2calo[ic] == int(ih)) {
        if (isEM[ic])
          sub += in.emcalo[ic].hwPt;
        else
          keep = true;
      }
    }
    dpt_t emdiff = dpt_t(in.hadcalo[ih].hwEmPt) - sub;  // ok to saturate at zero here
    dpt_t alldiff = dpt_t(in.hadcalo[ih].hwPt) - sub;
    if (debug_ && (in.hadcalo[ih].hwPt > 0)) {
      printf("FW  \t calo   %3d pt %8.2f has a subtracted pt of %8.2f, empt %8.2f -> %8.2f   isem %d mustkeep %d \n",
             ih,
             in.hadcalo[ih].floatPt(),
             Scales::floatPt(alldiff),
             in.hadcalo[ih].floatEmPt(),
             Scales::floatPt(emdiff),
             int(in.hadcalo[ih].hwIsEM),
             keep);
    }
    if (alldiff <= (in.hadcalo[ih].hwPt >> 4)) {
      hadcalo_out[ih].hwPt = 0;    // kill
      hadcalo_out[ih].hwEmPt = 0;  // kill
      if (debug_ && (in.hadcalo[ih].hwPt > 0))
        printf("FW  \t calo   %3d pt %8.2f --> discarded (zero pt)\n", ih, in.hadcalo[ih].floatPt());
    } else if ((in.hadcalo[ih].hwIsEM && emdiff <= (in.hadcalo[ih].hwEmPt >> 3)) && !keep) {
      hadcalo_out[ih].hwPt = 0;    // kill
      hadcalo_out[ih].hwEmPt = 0;  // kill
      if (debug_ && (in.hadcalo[ih].hwPt > 0))
        printf("FW  \t calo   %3d pt %8.2f --> discarded (zero em)\n", ih, in.hadcalo[ih].floatPt());
    } else {
      hadcalo_out[ih].hwPt = alldiff;
      hadcalo_out[ih].hwEmPt = (emdiff > 0 ? pt_t(emdiff) : pt_t(0));
    }
  }
}

void l1ct::PFAlgo3Emulator::run(const PFInputRegion& in, OutputRegion& out) const {
  // constants
  unsigned int nTRACK = std::min<unsigned>(nTRACK_, in.track.size());
  unsigned int nEMCALO = std::min<unsigned>(nEMCALO_, in.emcalo.size());
  unsigned int nPHOTON = std::min<unsigned>(nPHOTON_, in.emcalo.size());
  unsigned int nCALO = std::min<unsigned>(nCALO_, in.hadcalo.size());
  unsigned int nSELCALO = std::min<unsigned>(nSELCALO_, in.hadcalo.size());
  unsigned int nMU = std::min<unsigned>(nMU_, in.muon.size());

  if (debug_) {
    printf("FW\nFW  \t region eta [ %+5.2f , %+5.2f ], phi [ %+5.2f , %+5.2f ]\n",
           in.region.floatEtaMinExtra(),
           in.region.floatEtaMaxExtra(),
           in.region.floatPhiCenter() - in.region.floatPhiHalfWidthExtra(),
           in.region.floatPhiCenter() + in.region.floatPhiHalfWidthExtra());

    printf("FW  \t N(track) %3lu   N(em) %3lu   N(calo) %3lu   N(mu) %3lu\n",
           in.track.size(),
           in.emcalo.size(),
           in.hadcalo.size(),
           in.muon.size());

    for (unsigned int i = 0; i < nTRACK; ++i) {
      if (in.track[i].hwPt == 0)
        continue;
      printf(
          "FW  \t track %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  vtx eta %+5.2f   "
          "vtx phi %+5.2f   charge %+2d  quality %d\n",
          i,
          in.track[i].floatPt(),
          in.track[i].intPt(),
          in.track[i].floatEta(),
          in.track[i].intEta(),
          in.track[i].floatPhi(),
          in.track[i].intPhi(),
          in.track[i].floatVtxEta(),
          in.track[i].floatVtxPhi(),
          in.track[i].intCharge(),
          int(in.track[i].hwQuality));
    }
    for (unsigned int i = 0; i < nEMCALO; ++i) {
      if (in.emcalo[i].hwPt == 0)
        continue;
      printf(
          "FW  \t EM    %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  calo ptErr %8.2f [ "
          "%6d ] \n",
          i,
          in.emcalo[i].floatPt(),
          in.emcalo[i].intPt(),
          in.emcalo[i].floatEta(),
          in.emcalo[i].intEta(),
          in.emcalo[i].floatPhi(),
          in.emcalo[i].intPhi(),
          in.emcalo[i].floatPtErr(),
          in.emcalo[i].intPtErr());
    }
    for (unsigned int i = 0; i < nCALO; ++i) {
      if (in.hadcalo[i].hwPt == 0)
        continue;
      printf(
          "FW  \t calo  %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  calo emPt  %8.2f [ "
          "%6d ]   isEM %d \n",
          i,
          in.hadcalo[i].floatPt(),
          in.hadcalo[i].intPt(),
          in.hadcalo[i].floatEta(),
          in.hadcalo[i].intEta(),
          in.hadcalo[i].floatPhi(),
          in.hadcalo[i].intPhi(),
          in.hadcalo[i].floatEmPt(),
          in.hadcalo[i].intEmPt(),
          int(in.hadcalo[i].hwIsEM));
    }
    for (unsigned int i = 0; i < nMU; ++i) {
      if (in.muon[i].hwPt == 0)
        continue;
      printf("FW  \t muon  %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  charge %+2d  \n",
             i,
             in.muon[i].floatPt(),
             in.muon[i].intPt(),
             in.muon[i].floatEta(),
             in.muon[i].intEta(),
             in.muon[i].floatPhi(),
             in.muon[i].intPhi(),
             in.muon[i].intCharge());
    }
    printf("FW\n");
  }

  ////////////////////////////////////////////////////
  // TK-MU Linking
  std::vector<int> iMu;
  pfalgo_mu_ref(in, out, iMu);

  ////////////////////////////////////////////////////
  // TK-EM Linking
  std::vector<int> iEle;
  std::vector<HadCaloObjEmu> hadcalo_subem(nCALO);
  pfalgo3_em_ref(in, iMu, iEle, out, hadcalo_subem);

  ////////////////////////////////////////////////////
  // TK-HAD Linking

  // initialize sum track pt
  std::vector<pt_t> calo_sumtk(nCALO), calo_subpt(nCALO);
  std::vector<pt2_t> calo_sumtkErr2(nCALO);
  for (unsigned int ic = 0; ic < nCALO; ++ic) {
    calo_sumtk[ic] = 0;
    calo_sumtkErr2[ic] = 0;
  }

  // initialize good track bit
  std::vector<bool> track_good(nTRACK, false);
  for (unsigned int it = 0; it < nTRACK; ++it) {
    if (!in.track[it].isPFLoose())
      continue;
    pt_t ptInv = in.track[it].isPFTight() ? tk_MAXINVPT_TIGHT_ : tk_MAXINVPT_LOOSE_;
    track_good[it] = (in.track[it].hwPt < ptInv) || (iEle[it] != -1) || (iMu[it] != -1);
  }

  // initialize output
  out.pfcharged.resize(nTRACK);
  out.pfneutral.resize(nSELCALO);
  for (unsigned int ipf = 0; ipf < nTRACK; ++ipf)
    out.pfcharged[ipf].clear();
  for (unsigned int ipf = 0; ipf < nSELCALO; ++ipf)
    out.pfneutral[ipf].clear();

  // for each track, find the closest calo
  std::vector<int> tk2calo(nTRACK, -1);
  for (unsigned int it = 0; it < nTRACK; ++it) {
    if (in.track[it].hwPt > 0 && in.track[it].isPFLoose() && (iEle[it] == -1) && (iMu[it] == -1)) {
      pt_t tkCaloPtErr = ptErr_ref(in.region, in.track[it]);
      int ibest = best_match_with_pt_ref(dR2MAX_TK_CALO_, hadcalo_subem, in.track[it], tkCaloPtErr);
      if (ibest != -1) {
        if (debug_)
          printf(
              "FW  \t track  %3d pt %8.2f matched to calo %3d pt %8.2f (int deltaR2 %d)\n",
              it,
              in.track[it].floatPt(),
              ibest,
              hadcalo_subem[ibest].floatPt(),
              dr2_int(in.track[it].hwEta, in.track[it].hwPhi, hadcalo_subem[ibest].hwEta, hadcalo_subem[ibest].hwPhi));
        track_good[it] = true;
        calo_sumtk[ibest] += in.track[it].hwPt;
        calo_sumtkErr2[ibest] += tkCaloPtErr * tkCaloPtErr;
      }
      tk2calo[it] = ibest;  // for emulator info
    }
  }

  for (unsigned int ic = 0; ic < nCALO; ++ic) {
    if (calo_sumtk[ic] > 0) {
      dpt_t ptdiff = dpt_t(hadcalo_subem[ic].hwPt) - dpt_t(calo_sumtk[ic]);
      pt2_t sigmamult = calo_sumtkErr2
          [ic];  // before we did (calo_sumtkErr2[ic] + (calo_sumtkErr2[ic] >> 1)); to multiply by 1.5 = sqrt(1.5)^2 ~ (1.2)^2
      if (debug_ && (hadcalo_subem[ic].hwPt > 0)) {
        printf(
            "FW  \t calo  %3d pt %8.2f [ %7d ] eta %+5.2f [ %+5d ] has a sum track pt %8.2f, difference %7.2f +- %.2f "
            "\n",
            ic,
            hadcalo_subem[ic].floatPt(),
            hadcalo_subem[ic].intPt(),
            hadcalo_subem[ic].floatEta(),
            hadcalo_subem[ic].intEta(),
            Scales::floatPt(calo_sumtk[ic]),
            Scales::floatPt(ptdiff),
            std::sqrt(Scales::floatPt(calo_sumtkErr2[ic])));
      }
      if (ptdiff > 0 && ptdiff * ptdiff > sigmamult) {
        calo_subpt[ic] = pt_t(ptdiff);
      } else {
        calo_subpt[ic] = 0;
      }
    } else {
      calo_subpt[ic] = hadcalo_subem[ic].hwPt;
    }
    if (debug_ && (hadcalo_subem[ic].hwPt > 0))
      printf(
          "FW  \t calo  %3d pt %8.2f ---> %8.2f \n", ic, hadcalo_subem[ic].floatPt(), Scales::floatPt(calo_subpt[ic]));
  }

  // copy out charged hadrons
  for (unsigned int it = 0; it < nTRACK; ++it) {
    if (track_good[it]) {
      fillPFCand(in.track[it], out.pfcharged[it], iMu[it] != -1, iEle[it] != -1);
      // extra emulator information
      if (iEle[it] != -1)
        out.pfcharged[it].srcCluster = in.emcalo[iEle[it]].src;
      if (iMu[it] != -1)
        out.pfcharged[it].srcMu = in.muon[iMu[it]].src;
    }
  }

  // copy out neutral hadrons
  std::vector<PFNeutralObjEmu> outne_all(nCALO);
  for (unsigned int ipf = 0; ipf < nCALO; ++ipf)
    outne_all[ipf].clear();
  for (unsigned int ic = 0; ic < nCALO; ++ic) {
    if (calo_subpt[ic] > 0) {
      fillPFCand(hadcalo_subem[ic], outne_all[ic]);
      outne_all[ic].hwPt = calo_subpt[ic];
      outne_all[ic].hwEmPt = hadcalo_subem[ic].hwIsEM ? calo_subpt[ic] : pt_t(0);  // FIXME
    }
  }

  ptsort_ref(nCALO, nSELCALO, outne_all, out.pfneutral);

  if (debug_) {
    printf("FW\n");
    for (unsigned int i = 0; i < nTRACK; ++i) {
      if (out.pfcharged[i].hwPt == 0)
        continue;
      printf("FW  \t outch %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  pid %d\n",
             i,
             out.pfcharged[i].floatPt(),
             out.pfcharged[i].intPt(),
             out.pfcharged[i].floatEta(),
             out.pfcharged[i].intEta(),
             out.pfcharged[i].floatPhi(),
             out.pfcharged[i].intPhi(),
             out.pfcharged[i].intId());
    }
    for (unsigned int i = 0; i < nPHOTON; ++i) {
      if (out.pfphoton[i].hwPt == 0)
        continue;
      printf("FW  \t outph %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  pid %d\n",
             i,
             out.pfphoton[i].floatPt(),
             out.pfphoton[i].intPt(),
             out.pfphoton[i].floatEta(),
             out.pfphoton[i].intEta(),
             out.pfphoton[i].floatPhi(),
             out.pfphoton[i].intPhi(),
             out.pfphoton[i].intId());
    }
    for (unsigned int i = 0; i < nSELCALO; ++i) {
      if (out.pfneutral[i].hwPt == 0)
        continue;
      printf("FW  \t outne %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  pid %d\n",
             i,
             out.pfneutral[i].floatPt(),
             out.pfneutral[i].intPt(),
             out.pfneutral[i].floatEta(),
             out.pfneutral[i].intEta(),
             out.pfneutral[i].floatPhi(),
             out.pfneutral[i].intPhi(),
             out.pfneutral[i].intId());
    }
    printf("FW\n");
  }
}

void l1ct::PFAlgo3Emulator::mergeNeutrals(OutputRegion& out) const {
  out.pfphoton.reserve(out.pfphoton.size() + out.pfneutral.size());
  out.pfphoton.insert(out.pfphoton.end(), out.pfneutral.begin(), out.pfneutral.end());
  out.pfphoton.swap(out.pfneutral);
  out.pfphoton.clear();
}
