#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo2hgc_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <memory>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::PFAlgo2HGCEmulator::PFAlgo2HGCEmulator(const edm::ParameterSet& iConfig)
    : PFAlgoEmulatorBase(iConfig.getParameter<uint32_t>("nTrack"),
                         iConfig.getParameter<uint32_t>("nCalo"),
                         iConfig.getParameter<uint32_t>("nMu"),
                         iConfig.getParameter<uint32_t>("nSelCalo"),
                         l1ct::Scales::makeDR2FromFloatDR(iConfig.getParameter<double>("trackMuDR")),
                         l1ct::Scales::makeDR2FromFloatDR(iConfig.getParameter<double>("trackCaloDR")),
                         l1ct::Scales::makePtFromFloat(iConfig.getParameter<double>("maxInvisiblePt")),
                         l1ct::Scales::makePtFromFloat(iConfig.getParameter<double>("tightTrackMaxInvisiblePt"))) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  loadPtErrBins(iConfig);
}

#endif

void l1ct::PFAlgo2HGCEmulator::toFirmware(const PFInputRegion& in,
                                          PFRegion& region,
                                          HadCaloObj calo[/*nCALO*/],
                                          TkObj track[/*nTRACK*/],
                                          MuObj mu[/*nMU*/]) const {
  region = in.region;
  l1ct::toFirmware(in.track, nTRACK_, track);
  l1ct::toFirmware(in.hadcalo, nCALO_, calo);
  l1ct::toFirmware(in.muon, nMU_, mu);
}

void l1ct::PFAlgo2HGCEmulator::toFirmware(const OutputRegion& out,
                                          PFChargedObj outch[/*nTRACK*/],
                                          PFNeutralObj outne[/*nSELCALO*/],
                                          PFChargedObj outmu[/*nMU*/]) const {
  l1ct::toFirmware(out.pfcharged, nTRACK_, outch);
  l1ct::toFirmware(out.pfneutral, nSELCALO_, outne);
  l1ct::toFirmware(out.pfmuon, nMU_, outmu);
}

void l1ct::PFAlgo2HGCEmulator::run(const PFInputRegion& in, OutputRegion& out) const {
  unsigned int nTRACK = std::min<unsigned>(nTRACK_, in.track.size());
  unsigned int nCALO = std::min<unsigned>(nCALO_, in.hadcalo.size());
  unsigned int nSELCALO = std::min<unsigned>(nSELCALO_, in.hadcalo.size());
  unsigned int nMU = std::min<unsigned>(nMU_, in.muon.size());

  if (debug_) {
    dbgPrintf("FW\nFW  \t region eta %+5.2f [ %+5.2f , %+5.2f ], phi %+5.2f [ %+5.2f , %+5.2f ]   packed %s\n",
              in.region.floatEtaCenter(),
              in.region.floatEtaMinExtra(),
              in.region.floatEtaMaxExtra(),
              in.region.floatPhiCenter(),
              in.region.floatPhiCenter() - in.region.floatPhiHalfWidthExtra(),
              in.region.floatPhiCenter() + in.region.floatPhiHalfWidthExtra(),
              in.region.pack().to_string(16).c_str());

    dbgPrintf("FW  \t N(track) %3lu   N(calo) %3lu   N(mu) %3lu\n", in.track.size(), in.hadcalo.size(), in.muon.size());

    for (unsigned int i = 0; i < nTRACK; ++i) {
      if (in.track[i].hwPt == 0)
        continue;
      dbgPrintf(
          "FW  \t track %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  vtx eta %+5.2f  "
          "vtx phi %+5.2f  charge %+2d  qual %d  fid %d  glb eta %+5.2f phi %+5.2f  packed %s\n",
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
          int(in.track[i].hwQuality),
          int(in.region.isFiducial(in.track[i].hwEta, in.track[i].hwPhi)),
          in.region.floatGlbEta(in.track[i].hwVtxEta()),
          in.region.floatGlbPhi(in.track[i].hwVtxPhi()),
          in.track[i].pack().to_string(16).c_str());
    }
    for (unsigned int i = 0; i < nCALO; ++i) {
      if (in.hadcalo[i].hwPt == 0)
        continue;
      dbgPrintf(
          "FW  \t calo  %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  calo emPt %8.2f [ "
          "%6d ]   emID %2d  fid %d  glb eta %+5.2f phi %+5.2f  packed %s \n",
          i,
          in.hadcalo[i].floatPt(),
          in.hadcalo[i].intPt(),
          in.hadcalo[i].floatEta(),
          in.hadcalo[i].intEta(),
          in.hadcalo[i].floatPhi(),
          in.hadcalo[i].intPhi(),
          in.hadcalo[i].floatEmPt(),
          in.hadcalo[i].intEmPt(),
          in.hadcalo[i].hwEmID.to_int(),
          int(in.region.isFiducial(in.hadcalo[i].hwEta, in.hadcalo[i].hwPhi)),
          in.region.floatGlbEtaOf(in.hadcalo[i]),
          in.region.floatGlbPhiOf(in.hadcalo[i]),
          in.hadcalo[i].pack().to_string(16).c_str());
    }
    for (unsigned int i = 0; i < nMU; ++i) {
      if (in.muon[i].hwPt == 0)
        continue;
      dbgPrintf(
          "FW  \t muon  %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  "
          "vtx eta %+5.2f  vtx phi %+5.2f  charge %+2d  qual %2d  glb eta %+5.2f phi %+5.2f  packed %s \n",
          i,
          in.muon[i].floatPt(),
          in.muon[i].intPt(),
          in.muon[i].floatEta(),
          in.muon[i].intEta(),
          in.muon[i].floatPhi(),
          in.muon[i].intPhi(),
          in.muon[i].floatVtxEta(),
          in.muon[i].floatVtxPhi(),
          in.muon[i].intCharge(),
          int(in.muon[i].hwQuality),
          in.region.floatGlbEta(in.muon[i].hwVtxEta()),
          in.region.floatGlbPhi(in.muon[i].hwVtxPhi()),
          in.muon[i].pack().to_string(16).c_str());
    }
  }

  ////////////////////////////////////////////////////
  // TK-MU Linking
  std::vector<int> iMu;
  pfalgo_mu_ref(in, out, iMu);

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
  std::vector<bool> isEle(nTRACK, false);
  for (unsigned int it = 0; it < nTRACK; ++it) {
    if (!in.track[it].isPFLoose())
      continue;
    pt_t ptInv = in.track[it].isPFTight() ? tk_MAXINVPT_TIGHT_ : tk_MAXINVPT_LOOSE_;
    track_good[it] = (in.track[it].hwPt < ptInv) || (iMu[it] != -1);
    isEle[it] = false;
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
    if (in.track[it].hwPt > 0 && in.track[it].isPFLoose() && iMu[it] == -1) {
      pt_t tkCaloPtErr = ptErr_ref(in.region, in.track[it]);
      int ibest = best_match_with_pt_ref(dR2MAX_TK_CALO_, in.hadcalo, in.track[it], tkCaloPtErr);
      if (ibest != -1) {
        if (debug_)
          dbgPrintf("FW  \t track  %3d pt %8.2f caloPtErr %6.2f matched to calo %3d pt %8.2f\n",
                    it,
                    in.track[it].floatPt(),
                    Scales::floatPt(tkCaloPtErr),
                    ibest,
                    in.hadcalo[ibest].floatPt());
        track_good[it] = true;
        isEle[it] = in.hadcalo[ibest].hwIsEM();
        calo_sumtk[ibest] += in.track[it].hwPt;
        calo_sumtkErr2[ibest] += tkCaloPtErr * tkCaloPtErr;
      }
      tk2calo[it] = ibest;  // for emulator info
    }
  }

  for (unsigned int ic = 0; ic < nCALO; ++ic) {
    if (calo_sumtk[ic] > 0) {
      pt_t ptdiff = in.hadcalo[ic].hwPt - calo_sumtk[ic];
      pt2_t sigmamult =
          calo_sumtkErr2[ic];  //  + (calo_sumtkErr2[ic] >> 1)); // this multiplies by 1.5 = sqrt(1.5)^2 ~ (1.2)^2
      if (debug_ && (in.hadcalo[ic].hwPt > 0)) {
        dbgPrintf(
            "FW  \t calo  %3d pt %8.2f [ %7d ] eta %+5.2f [ %+5d ] has a sum track pt %8.2f, difference %7.2f +- %.2f "
            "\n",
            ic,
            in.hadcalo[ic].floatPt(),
            in.hadcalo[ic].intPt(),
            in.hadcalo[ic].floatEta(),
            in.hadcalo[ic].intEta(),
            Scales::floatPt(calo_sumtk[ic]),
            Scales::floatPt(ptdiff),
            std::sqrt(Scales::floatPt(calo_sumtkErr2[ic])));
      }
      if (ptdiff > 0 && ptdiff * ptdiff > sigmamult) {
        calo_subpt[ic] = ptdiff;
      } else {
        calo_subpt[ic] = 0;
      }
    } else {
      calo_subpt[ic] = in.hadcalo[ic].hwPt;
    }
    if (debug_ && (in.hadcalo[ic].hwPt > 0))
      dbgPrintf(
          "FW  \t calo'  %3d pt %8.2f ---> %8.2f \n", ic, in.hadcalo[ic].floatPt(), Scales::floatPt(calo_subpt[ic]));
  }

  // copy out charged hadrons
  for (unsigned int it = 0; it < nTRACK; ++it) {
    if (in.track[it].hwPt > 0 && track_good[it]) {
      fillPFCand(in.track[it], out.pfcharged[it], /*isMu=*/(iMu[it] != -1), isEle[it]);
      // extra emulator information
      if (tk2calo[it] != -1)
        out.pfcharged[it].srcCluster = in.hadcalo[tk2calo[it]].src;
      if (iMu[it] != -1)
        out.pfcharged[it].srcMu = in.muon[iMu[it]].src;
    }
  }

  // copy out neutral hadrons with sorting and cropping
  std::vector<PFNeutralObjEmu> outne_all(nCALO);
  for (unsigned int ipf = 0; ipf < nCALO; ++ipf)
    outne_all[ipf].clear();
  for (unsigned int ic = 0; ic < nCALO; ++ic) {
    if (calo_subpt[ic] > 0) {
      fillPFCand(in.hadcalo[ic], outne_all[ic], in.hadcalo[ic].hwIsEM());
      outne_all[ic].hwPt = calo_subpt[ic];
      outne_all[ic].hwEmPt = in.hadcalo[ic].hwIsEM() ? calo_subpt[ic] : pt_t(0);  // FIXME
    }
  }

  if (nCALO_ == nSELCALO_) {
    std::swap(outne_all, out.pfneutral);
  } else {
    ptsort_ref(nCALO, nSELCALO, outne_all, out.pfneutral);
  }

  if (debug_) {
    for (unsigned int i = 0; i < nTRACK; ++i) {
      if (out.pfcharged[i].hwPt == 0)
        continue;
      dbgPrintf(
          "FW  \t outch %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  pid %d  packed %s\n",
          i,
          out.pfcharged[i].floatPt(),
          out.pfcharged[i].intPt(),
          out.pfcharged[i].floatEta(),
          out.pfcharged[i].intEta(),
          out.pfcharged[i].floatPhi(),
          out.pfcharged[i].intPhi(),
          out.pfcharged[i].intId(),
          out.pfcharged[i].pack().to_string(16).c_str());
    }
    for (unsigned int i = 0; i < nSELCALO; ++i) {
      if (out.pfneutral[i].hwPt == 0)
        continue;
      dbgPrintf(
          "FW  \t outne %3d: pt %8.2f [ %8d ]  calo eta %+5.2f [ %+5d ]  calo phi %+5.2f [ %+5d ]  pid %d  packed %s\n",
          i,
          out.pfneutral[i].floatPt(),
          out.pfneutral[i].intPt(),
          out.pfneutral[i].floatEta(),
          out.pfneutral[i].intEta(),
          out.pfneutral[i].floatPhi(),
          out.pfneutral[i].intPhi(),
          out.pfneutral[i].intId(),
          out.pfneutral[i].pack().to_string(16).c_str());
    }
  }
}
