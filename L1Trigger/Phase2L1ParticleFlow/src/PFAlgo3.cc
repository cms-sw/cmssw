#include "L1Trigger/Phase2L1ParticleFlow/interface/PFAlgo3.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Math/interface/deltaR.h"

namespace {
  template <typename T1, typename T2>
  float floatDR(const T1 &t1, const T2 &t2) {
    return deltaR(t1.floatEta(), t1.floatPhi(), t2.floatEta(), t2.floatPhi());
  }
}  // namespace

using namespace l1tpf_impl;

PFAlgo3::PFAlgo3(const edm::ParameterSet &iConfig) : PFAlgoBase(iConfig) {
  debug_ = iConfig.getUntrackedParameter<int>("debugPFAlgo3", iConfig.getUntrackedParameter<int>("debug", 0));
  edm::ParameterSet linkcfg = iConfig.getParameter<edm::ParameterSet>("linking");
  drMatchMu_ = linkcfg.getParameter<double>("trackMuDR");

  std::string muMatchMode = linkcfg.getParameter<std::string>("trackMuMatch");
  if (muMatchMode == "boxBestByPtRatio")
    muMatchMode_ = MuMatchMode::BoxBestByPtRatio;
  else if (muMatchMode == "drBestByPtRatio")
    muMatchMode_ = MuMatchMode::DrBestByPtRatio;
  else if (muMatchMode == "drBestByPtDiff")
    muMatchMode_ = MuMatchMode::DrBestByPtDiff;
  else
    throw cms::Exception("Configuration", "bad value for trackMuMatch configurable");

  std::string tkCaloLinkMetric = linkcfg.getParameter<std::string>("trackCaloLinkMetric");
  if (tkCaloLinkMetric == "bestByDR")
    tkCaloLinkMetric_ = TkCaloLinkMetric::BestByDR;
  else if (tkCaloLinkMetric == "bestByDRPt")
    tkCaloLinkMetric_ = TkCaloLinkMetric::BestByDRPt;
  else if (tkCaloLinkMetric == "bestByDR2Pt2")
    tkCaloLinkMetric_ = TkCaloLinkMetric::BestByDR2Pt2;
  else
    throw cms::Exception("Configuration", "bad value for tkCaloLinkMetric configurable");

  drMatch_ = linkcfg.getParameter<double>("trackCaloDR");
  ptMatchLow_ = linkcfg.getParameter<double>("trackCaloNSigmaLow");
  ptMatchHigh_ = linkcfg.getParameter<double>("trackCaloNSigmaHigh");
  useTrackCaloSigma_ = linkcfg.getParameter<bool>("useTrackCaloSigma");
  maxInvisiblePt_ = linkcfg.getParameter<double>("maxInvisiblePt");

  drMatchEm_ = linkcfg.getParameter<double>("trackEmDR");
  trackEmUseAlsoTrackSigma_ = linkcfg.getParameter<bool>("trackEmUseAlsoTrackSigma");
  trackEmMayUseCaloMomenta_ = linkcfg.getParameter<bool>("trackEmMayUseCaloMomenta");
  emCaloUseAlsoCaloSigma_ = linkcfg.getParameter<bool>("emCaloUseAlsoCaloSigma");
  ptMinFracMatchEm_ = linkcfg.getParameter<double>("caloEmPtMinFrac");
  drMatchEmHad_ = linkcfg.getParameter<double>("emCaloDR");
  emHadSubtractionPtSlope_ = linkcfg.getParameter<double>("emCaloSubtractionPtSlope");
  caloReLinkStep_ = linkcfg.getParameter<bool>("caloReLink");
  caloReLinkDr_ = linkcfg.getParameter<double>("caloReLinkDR");
  caloReLinkThreshold_ = linkcfg.getParameter<double>("caloReLinkThreshold");
  rescaleTracks_ = linkcfg.getParameter<bool>("rescaleTracks");
  caloTrkWeightedAverage_ = linkcfg.getParameter<bool>("useCaloTrkWeightedAverage");
  sumTkCaloErr2_ = linkcfg.getParameter<bool>("sumTkCaloErr2");
  ecalPriority_ = linkcfg.getParameter<bool>("ecalPriority");
  tightTrackMaxInvisiblePt_ = linkcfg.getParameter<double>("tightTrackMaxInvisiblePt");
  sortInputs_ = iConfig.getParameter<bool>("sortInputs");
}

void PFAlgo3::runPF(Region &r) const {
  initRegion(r, sortInputs_);

  /// ------------- first step (can all go in parallel) ----------------

  if (debug_) {
    dbgPrintf(
        "PFAlgo3\nPFAlgo3 region eta [ %+5.2f , %+5.2f ], phi [ %+5.2f , %+5.2f ], fiducial eta [ %+5.2f , %+5.2f ], "
        "phi [ %+5.2f , %+5.2f ]\n",
        r.etaMin - r.etaExtra,
        r.etaMax + r.etaExtra,
        r.phiCenter - r.phiHalfWidth - r.phiExtra,
        r.phiCenter + r.phiHalfWidth + r.phiExtra,
        r.etaMin,
        r.etaMax,
        r.phiCenter - r.phiHalfWidth,
        r.phiCenter + r.phiHalfWidth);
    dbgPrintf("PFAlgo3 \t N(track) %3lu   N(em) %3lu   N(calo) %3lu   N(mu) %3lu\n",
              r.track.size(),
              r.emcalo.size(),
              r.calo.size(),
              r.muon.size());
    for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
      const auto &tk = r.track[itk];
      dbgPrintf(
          "PFAlgo3 \t track %3d: pt %7.2f +- %5.2f  vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f  "
          "fid %1d  calo ptErr %7.2f stubs %2d chi2 %7.1f quality %d\n",
          itk,
          tk.floatPt(),
          tk.floatPtErr(),
          tk.floatVtxEta(),
          tk.floatVtxPhi(),
          tk.floatEta(),
          tk.floatPhi(),
          int(r.fiducialLocal(tk.floatEta(), tk.floatPhi())),
          tk.floatCaloPtErr(),
          int(tk.hwStubs),
          tk.hwChi2 * 0.1f,
          int(tk.hwFlags));
    }
    for (int iem = 0, nem = r.emcalo.size(); iem < nem; ++iem) {
      const auto &em = r.emcalo[iem];
      dbgPrintf(
          "PFAlgo3 \t EM    %3d: pt %7.2f +- %5.2f  vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f  "
          "fid %1d  calo ptErr %7.2f\n",
          iem,
          em.floatPt(),
          em.floatPtErr(),
          em.floatEta(),
          em.floatPhi(),
          em.floatEta(),
          em.floatPhi(),
          int(r.fiducialLocal(em.floatEta(), em.floatPhi())),
          em.floatPtErr());
    }
    for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
      auto &calo = r.calo[ic];
      dbgPrintf(
          "PFAlgo3 \t calo  %3d: pt %7.2f +- %5.2f  vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f  "
          "fid %1d  calo ptErr %7.2f em pt %7.2f \n",
          ic,
          calo.floatPt(),
          calo.floatPtErr(),
          calo.floatEta(),
          calo.floatPhi(),
          calo.floatEta(),
          calo.floatPhi(),
          int(r.fiducialLocal(calo.floatEta(), calo.floatPhi())),
          calo.floatPtErr(),
          calo.floatEmPt());
    }
    for (int im = 0, nm = r.muon.size(); im < nm; ++im) {
      auto &mu = r.muon[im];
      dbgPrintf(
          "PFAlgo3 \t muon  %3d: pt %7.2f           vtx eta %+5.2f  vtx phi %+5.2f  calo eta %+5.2f  calo phi %+5.2f  "
          "fid %1d \n",
          im,
          mu.floatPt(),
          mu.floatEta(),
          mu.floatPhi(),
          mu.floatEta(),
          mu.floatPhi(),
          int(r.fiducialLocal(mu.floatEta(), mu.floatPhi())));
    }
  }

  std::vector<int> tk2mu(r.track.size(), -1), mu2tk(r.muon.size(), -1);
  link_tk2mu(r, tk2mu, mu2tk);

  // match all tracks to the closest EM cluster
  std::vector<int> tk2em(r.track.size(), -1);
  link_tk2em(r, tk2em);

  // match all em to the closest had (can happen in parallel to the above)
  std::vector<int> em2calo(r.emcalo.size(), -1);
  link_em2calo(r, em2calo);

  /// ------------- next step (needs the previous) ----------------
  // for each EM cluster, count and add up the pt of all the corresponding tracks (skipping muons)
  std::vector<int> em2ntk(r.emcalo.size(), 0);
  std::vector<float> em2sumtkpt(r.emcalo.size(), 0);
  std::vector<float> em2sumtkpterr(r.emcalo.size(), 0);
  sum_tk2em(r, tk2em, em2ntk, em2sumtkpt, em2sumtkpterr);

  /// ------------- next step (needs the previous) ----------------
  // process ecal clusters after linking
  emcalo_algo(r, em2ntk, em2sumtkpt, em2sumtkpterr);

  /// ------------- next step (needs the previous) ----------------
  // promote all flagged tracks to electrons
  emtk_algo(r, tk2em, em2ntk, em2sumtkpterr);
  sub_em2calo(r, em2calo);

  /// ------------- next step (needs the previous) ----------------
  // track to calo matching (first iteration, with a lower bound on the calo pt; there may be another one later)
  std::vector<int> tk2calo(r.track.size(), -1);
  link_tk2calo(r, tk2calo);

  /// ------------- next step (needs the previous) ----------------
  // for each calo, compute the sum of the track pt
  std::vector<int> calo2ntk(r.calo.size(), 0);
  std::vector<float> calo2sumtkpt(r.calo.size(), 0);
  std::vector<float> calo2sumtkpterr(r.calo.size(), 0);
  sum_tk2calo(r, tk2calo, calo2ntk, calo2sumtkpt, calo2sumtkpterr);

  // in the meantime, promote unlinked low pt tracks to hadrons
  unlinkedtk_algo(r, tk2calo);

  /// ------------- next step (needs the previous) ----------------
  /// OPTIONAL STEP: try to recover split hadron showers (v1.0):
  //     off by default, as it seems to not do much in jets even if it helps remove tails in single-pion events
  if (caloReLinkStep_)
    calo_relink(r, calo2ntk, calo2sumtkpt, calo2sumtkpterr);

  /// ------------- next step (needs the previous) ----------------
  // process matched calo clusters, compare energy to sum track pt
  std::vector<float> calo2alpha(r.calo.size(), 1);
  linkedcalo_algo(r, calo2ntk, calo2sumtkpt, calo2sumtkpterr, calo2alpha);

  /// ------------- next step (needs the previous) ----------------
  /// process matched tracks, if necessary rescale or average
  linkedtk_algo(r, tk2calo, calo2ntk, calo2alpha);
  // process unmatched calo clusters
  unlinkedcalo_algo(r);
  // finally do muons
  save_muons(r, tk2mu);
}

void PFAlgo3::link_tk2mu(Region &r, std::vector<int> &tk2mu, std::vector<int> &mu2tk) const {
  // do a rectangular match for the moment; make a box of the same are as a 0.2 cone
  int intDrMuonMatchBox = std::ceil(drMatchMu_ * CaloCluster::ETAPHI_SCALE * std::sqrt(M_PI / 4));
  for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
    tk2mu[itk] = false;
  }
  for (int imu = 0, nmu = r.muon.size(); imu < nmu; ++imu) {
    const auto &mu = r.muon[imu];
    if (debug_)
      dbgPrintf(
          "PFAlgo3 \t muon  %3d (pt %7.2f, eta %+5.2f, phi %+5.2f) \n", imu, mu.floatPt(), mu.floatEta(), mu.floatPhi());
    float minDistance = 9e9;
    switch (muMatchMode_) {
      case MuMatchMode::BoxBestByPtRatio:
        minDistance = 4.;
        break;
      case MuMatchMode::DrBestByPtRatio:
        minDistance = 4.;
        break;
      case MuMatchMode::DrBestByPtDiff:
        minDistance = 0.5 * mu.floatPt();
        break;
    }
    int imatch = -1;
    for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
      const auto &tk = r.track[itk];
      if (!tk.quality(l1tpf_impl::InputTrack::PFLOOSE))
        continue;
      int deta = std::abs(mu.hwEta - tk.hwEta);
      int dphi = std::abs((mu.hwPhi - tk.hwPhi) % CaloCluster::PHI_WRAP);
      float dr = floatDR(mu, tk);
      float dpt = std::abs(mu.floatPt() - tk.floatPt());
      float dptr = (mu.hwPt > tk.hwPt ? mu.floatPt() / tk.floatPt() : tk.floatPt() / mu.floatPt());
      bool ok = false;
      float distance = 9e9;
      switch (muMatchMode_) {
        case MuMatchMode::BoxBestByPtRatio:
          ok = (deta < intDrMuonMatchBox) && (dphi < intDrMuonMatchBox);
          distance = dptr;
          break;
        case MuMatchMode::DrBestByPtRatio:
          ok = (dr < drMatchMu_);
          distance = dptr;
          break;
        case MuMatchMode::DrBestByPtDiff:
          ok = (dr < drMatchMu_);
          distance = dpt;
          break;
      }
      if (debug_ && dr < 0.4) {
        dbgPrintf(
            "PFAlgo3 \t\t possible match with track %3d (pt %7.2f, caloeta %+5.2f, calophi %+5.2f, dr %.2f, eta "
            "%+5.2f, phi %+5.2f, dr %.2f):  angular %1d, distance %.3f (vs %.3f)\n",
            itk,
            tk.floatPt(),
            tk.floatEta(),
            tk.floatPhi(),
            dr,
            tk.floatVtxEta(),
            tk.floatVtxPhi(),
            deltaR(mu.floatEta(), mu.floatPhi(), tk.floatVtxEta(), tk.floatVtxPhi()),
            (ok ? 1 : 0),
            distance,
            minDistance);
      }
      if (!ok)
        continue;
      // FIXME for the moment, we do the floating point matching in pt
      if (distance < minDistance) {
        minDistance = distance;
        imatch = itk;
      }
    }
    if (debug_ && imatch > -1)
      dbgPrintf("PFAlgo3 \t muon  %3d (pt %7.2f) linked to track %3d (pt %7.2f)\n",
                imu,
                mu.floatPt(),
                imatch,
                r.track[imatch].floatPt());
    if (debug_ && imatch == -1)
      dbgPrintf("PFAlgo3 \t muon  %3d (pt %7.2f) not linked to any track\n", imu, mu.floatPt());
    mu2tk[imu] = imatch;
    if (imatch > -1) {
      tk2mu[imatch] = imu;
      r.track[imatch].muonLink = true;
    }
  }
}

void PFAlgo3::link_tk2em(Region &r, std::vector<int> &tk2em) const {
  // match all tracks to the closest EM cluster
  for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
    const auto &tk = r.track[itk];
    if (!tk.quality(l1tpf_impl::InputTrack::PFLOOSE))
      continue;
    //if (tk.muonLink) continue; // not necessary I think
    float drbest = drMatchEm_;
    for (int iem = 0, nem = r.emcalo.size(); iem < nem; ++iem) {
      const auto &em = r.emcalo[iem];
      float dr = floatDR(tk, em);
      if (dr < drbest) {
        tk2em[itk] = iem;
        drbest = dr;
      }
    }
    if (debug_ && tk2em[itk] != -1)
      dbgPrintf("PFAlgo3 \t track %3d (pt %7.2f) matches to EM   %3d (pt %7.2f) with dr %.3f\n",
                itk,
                tk.floatPt(),
                tk2em[itk],
                tk2em[itk] == -1 ? 0.0 : r.emcalo[tk2em[itk]].floatPt(),
                drbest);
  }
}

void PFAlgo3::link_em2calo(Region &r, std::vector<int> &em2calo) const {
  // match all em to the closest had (can happen in parallel to the above)
  for (int iem = 0, nem = r.emcalo.size(); iem < nem; ++iem) {
    const auto &em = r.emcalo[iem];
    float drbest = drMatchEmHad_;
    for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
      const auto &calo = r.calo[ic];
      if (calo.floatEmPt() < ptMinFracMatchEm_ * em.floatPt())
        continue;
      float dr = floatDR(calo, em);
      if (dr < drbest) {
        em2calo[iem] = ic;
        drbest = dr;
      }
    }
    if (debug_ && em2calo[iem] != -1)
      dbgPrintf("PFAlgo3 \t EM    %3d (pt %7.2f) matches to calo %3d (pt %7.2f, empt %7.2f) with dr %.3f\n",
                iem,
                em.floatPt(),
                em2calo[iem],
                em2calo[iem] == -1 ? 0.0 : r.calo[em2calo[iem]].floatPt(),
                em2calo[iem] == -1 ? 0.0 : r.calo[em2calo[iem]].floatEmPt(),
                drbest);
  }
}

void PFAlgo3::sum_tk2em(Region &r,
                        const std::vector<int> &tk2em,
                        std::vector<int> &em2ntk,
                        std::vector<float> &em2sumtkpt,
                        std::vector<float> &em2sumtkpterr) const {
  // for each EM cluster, count and add up the pt of all the corresponding tracks (skipping muons)
  for (int iem = 0, nem = r.emcalo.size(); iem < nem; ++iem) {
    const auto &em = r.emcalo[iem];
    if (r.globalAbsEta(em.floatEta()) > 2.5)
      continue;
    for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
      if (tk2em[itk] == iem) {
        const auto &tk = r.track[itk];
        if (tk.muonLink)
          continue;
        em2ntk[iem]++;
        em2sumtkpt[iem] += tk.floatPt();
        em2sumtkpterr[iem] += tk.floatPtErr();
      }
    }
  }
}

void PFAlgo3::emcalo_algo(Region &r,
                          const std::vector<int> &em2ntk,
                          const std::vector<float> &em2sumtkpt,
                          const std::vector<float> &em2sumtkpterr) const {
  // process ecal clusters after linking
  for (int iem = 0, nem = r.emcalo.size(); iem < nem; ++iem) {
    auto &em = r.emcalo[iem];
    em.isEM = false;
    em.used = false;
    em.hwFlags = 0;
    if (r.globalAbsEta(em.floatEta()) > 2.5)
      continue;
    if (debug_)
      dbgPrintf("PFAlgo3 \t EM    %3d (pt %7.2f) has %2d tracks (sumpt %7.2f, sumpterr %7.2f), ptdif %7.2f +- %7.2f\n",
                iem,
                em.floatPt(),
                em2ntk[iem],
                em2sumtkpt[iem],
                em2sumtkpterr[iem],
                em.floatPt() - em2sumtkpt[iem],
                std::max<float>(em2sumtkpterr[iem], em.floatPtErr()));
    if (em2ntk[iem] == 0) {  // Photon
      em.isEM = true;
      addCaloToPF(r, em);
      em.used = true;
      if (debug_)
        dbgPrintf("PFAlgo3 \t EM    %3d (pt %7.2f)    ---> promoted to photon\n", iem, em.floatPt());
      continue;
    }
    float ptdiff = em.floatPt() - em2sumtkpt[iem];
    float pterr = trackEmUseAlsoTrackSigma_ ? std::max<float>(em2sumtkpterr[iem], em.floatPtErr()) : em.floatPtErr();
    // avoid "pt = inf +- inf" track to become an electron.
    if (pterr > 2 * em.floatPt()) {
      pterr = 2 * em.floatPt();
      if (debug_)
        dbgPrintf("PFAlgo3 \t EM    %3d (pt %7.2f)    ---> clamp pterr ---> new ptdiff %7.2f +- %7.2f\n",
                  iem,
                  em.floatPt(),
                  ptdiff,
                  pterr);
    }

    if (ptdiff > -ptMatchLow_ * pterr) {
      em.isEM = true;
      em.used = true;
      // convert leftover to a photon if significant
      if (ptdiff > +ptMatchHigh_ * pterr) {
        auto &p = addCaloToPF(r, em);
        p.setFloatPt(ptdiff);
        if (debug_)
          dbgPrintf("PFAlgo3 \t EM    %3d (pt %7.2f)    ---> promoted to electron(s) + photon (pt %7.2f)\n",
                    iem,
                    em.floatPt(),
                    ptdiff);
      } else {
        em.hwFlags = 1;  // may use calo momentum
        if (debug_)
          dbgPrintf("PFAlgo3 \t EM    %3d (pt %7.2f)    ---> promoted to electron(s)\n", iem, em.floatPt());
      }
    } else {
      em.isEM = false;
      em.used = false;
      em.hwFlags = 0;
      //discardCalo(r, em, 2);
    }
  }
}

void PFAlgo3::emtk_algo(Region &r,
                        const std::vector<int> &tk2em,
                        const std::vector<int> &em2ntk,
                        const std::vector<float> &em2sumtkpterr) const {
  // promote all flagged tracks to electrons
  for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
    auto &tk = r.track[itk];
    if (tk2em[itk] == -1 || tk.muonLink)
      continue;
    const auto &em = r.emcalo[tk2em[itk]];
    if (em.isEM) {
      auto &p = addTrackToPF(r, tk);
      p.cluster.src = em.src;
      // FIXME to check if this is useful
      if (trackEmMayUseCaloMomenta_ && em2ntk[tk2em[itk]] == 1 && em.hwFlags == 1) {
        if (em.floatPtErr() < em2sumtkpterr[tk2em[itk]]) {
          p.setFloatPt(em.floatPt());
        }
      }
      if (debug_)
        dbgPrintf("PFAlgo3 \t track %3d (pt %7.2f) matched to EM   %3d (pt %7.2f) promoted to electron with pt %7.2f\n",
                  itk,
                  tk.floatPt(),
                  tk2em[itk],
                  em.floatPt(),
                  p.floatPt());
      p.hwId = l1t::PFCandidate::Electron;
      tk.used = true;
    }
  }
}

void PFAlgo3::sub_em2calo(Region &r, const std::vector<int> &em2calo) const {
  // subtract EM component from Calo clusters for all photons and electrons (within tracker coverage)
  // kill clusters that end up below their own uncertainty, or that loose 90% of the energy,
  // unless they still have live EM clusters pointing to them
  for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
    auto &calo = r.calo[ic];
    float pt0 = calo.floatPt(), ept0 = calo.floatEmPt(), pt = pt0, ept = ept0;
    bool keepme = false;
    for (int iem = 0, nem = r.emcalo.size(); iem < nem; ++iem) {
      if (em2calo[iem] == ic) {
        const auto &em = r.emcalo[iem];
        if (em.isEM) {
          if (debug_)
            dbgPrintf(
                "PFAlgo3 \t EM    %3d (pt %7.2f) is  subtracted from calo %3d (pt %7.2f) scaled by %.3f (deltaPt = "
                "%7.2f)\n",
                iem,
                em.floatPt(),
                ic,
                calo.floatPt(),
                emHadSubtractionPtSlope_,
                emHadSubtractionPtSlope_ * em.floatPt());
          pt -= emHadSubtractionPtSlope_ * em.floatPt();
          ept -= em.floatPt();
        } else {
          keepme = true;
          if (debug_)
            dbgPrintf(
                "PFAlgo3 \t EM    %3d (pt %7.2f) not subtracted from calo %3d (pt %7.2f), and calo marked to be kept "
                "after EM subtraction\n",
                iem,
                em.floatPt(),
                ic,
                calo.floatPt());
        }
      }
    }
    if (pt < pt0) {
      if (debug_)
        dbgPrintf(
            "PFAlgo3 \t calo  %3d (pt %7.2f +- %7.2f) has a subtracted pt of %7.2f, empt %7.2f -> %7.2f, isem %d\n",
            ic,
            calo.floatPt(),
            calo.floatPtErr(),
            pt,
            ept0,
            ept,
            calo.isEM);
      calo.setFloatPt(pt);
      calo.setFloatEmPt(ept);
      if (!keepme &&
          ((emCaloUseAlsoCaloSigma_ ? pt < calo.floatPtErr() : false) || pt <= 0.125 * pt0 ||
           (calo.isEM && ept <= 0.125 * ept0))) {  // the <= is important since in firmware the pt0/8 can be zero
        if (debug_)
          dbgPrintf("PFAlgo3 \t calo  %3d (pt %7.2f)    ----> discarded\n", ic, calo.floatPt());
        calo.used = true;
        calo.setFloatPt(pt0);  //discardCalo(r, calo, 1);  // log this as discarded, for debugging
      }
    }
  }
}

void PFAlgo3::link_tk2calo(Region &r, std::vector<int> &tk2calo) const {
  // track to calo matching (first iteration, with a lower bound on the calo pt; there may be another one later)
  for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
    const auto &tk = r.track[itk];
    if (!tk.quality(l1tpf_impl::InputTrack::PFLOOSE))
      continue;
    if (tk.muonLink || tk.used)
      continue;  // not necessary but just a waste of CPU otherwise
    float drbest = drMatch_, dptscale = 0;
    switch (tkCaloLinkMetric_) {
      case TkCaloLinkMetric::BestByDR:
        drbest = drMatch_;
        break;
      case TkCaloLinkMetric::BestByDRPt:
        drbest = 1.0;
        dptscale = drMatch_ / tk.floatCaloPtErr();
        break;
      case TkCaloLinkMetric::BestByDR2Pt2:
        drbest = 1.0;
        dptscale = drMatch_ / tk.floatCaloPtErr();
        break;
    }
    float minCaloPt = tk.floatPt() - ptMatchLow_ * tk.floatCaloPtErr();
    if (debug_)
      dbgPrintf("PFAlgo3 \t track %3d (pt %7.2f) to be matched to calo, min pT %7.2f\n", itk, tk.floatPt(), minCaloPt);
    for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
      auto &calo = r.calo[ic];
      if (calo.used || calo.floatPt() <= minCaloPt)
        continue;
      float dr = floatDR(tk, calo), dq;
      switch (tkCaloLinkMetric_) {
        case TkCaloLinkMetric::BestByDR:
          if (dr < drbest) {
            tk2calo[itk] = ic;
            drbest = dr;
          }
          break;
        case TkCaloLinkMetric::BestByDRPt:
          dq = dr + std::max<float>(tk.floatPt() - calo.floatPt(), 0.) * dptscale;
          //if (debug_ && dr < 0.2) dbgPrintf("PFAlgo3 \t\t\t track %3d (pt %7.2f) vs calo %3d (pt %7.2f): dr %.3f, dq %.3f\n", itk, tk.floatPt(), ic, calo.floatPt(), dr, dq);
          if (dr < drMatch_ && dq < drbest) {
            tk2calo[itk] = ic;
            drbest = dq;
          }
          break;
        case TkCaloLinkMetric::BestByDR2Pt2:
          dq = hypot(dr, std::max<float>(tk.floatPt() - calo.floatPt(), 0.) * dptscale);
          //if (debug_ && dr < 0.2) dbgPrintf("PFAlgo3 \t\t\t track %3d (pt %7.2f) vs calo %3d (pt %7.2f): dr %.3f, dq %.3f\n", itk, tk.floatPt(), ic, calo.floatPt(), dr, dq);
          if (dr < drMatch_ && dq < drbest) {
            tk2calo[itk] = ic;
            drbest = dq;
          }
          break;
      }
    }
    if (debug_ && tk2calo[itk] != -1)
      dbgPrintf("PFAlgo3 \t track %3d (pt %7.2f) matches to calo %3d (pt %7.2f) with dist %.3f\n",
                itk,
                tk.floatPt(),
                tk2calo[itk],
                tk2calo[itk] == -1 ? 0.0 : r.calo[tk2calo[itk]].floatPt(),
                drbest);
    // now we re-do this for debugging sake, it may be done for real later
    if (debug_ && tk2calo[itk] == -1) {
      int ibest = -1;
      drbest = 0.3;
      for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
        auto &calo = r.calo[ic];
        if (calo.used)
          continue;
        float dr = floatDR(tk, calo);
        if (dr < drbest) {
          ibest = ic;
          drbest = dr;
        }
      }
      if (ibest != -1)
        dbgPrintf(
            "PFAlgo3 \t track %3d (pt %7.2f) would match to calo %3d (pt %7.2f) with dr %.3f if the pt min and dr "
            "requirement had been relaxed\n",
            itk,
            tk.floatPt(),
            ibest,
            r.calo[ibest].floatPt(),
            drbest);
    }
  }
}

void PFAlgo3::sum_tk2calo(Region &r,
                          const std::vector<int> &tk2calo,
                          std::vector<int> &calo2ntk,
                          std::vector<float> &calo2sumtkpt,
                          std::vector<float> &calo2sumtkpterr) const {
  // for each calo, compute the sum of the track pt
  for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
    const auto &calo = r.calo[ic];
    if (r.globalAbsEta(calo.floatEta()) > 2.5)
      continue;
    for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
      if (tk2calo[itk] == ic) {
        const auto &tk = r.track[itk];
        if (tk.muonLink || tk.used)
          continue;
        calo2ntk[ic]++;
        calo2sumtkpt[ic] += tk.floatPt();
        calo2sumtkpterr[ic] += std::pow(tk.floatCaloPtErr(), sumTkCaloErr2_ ? 2 : 1);
      }
    }
    if (sumTkCaloErr2_ && calo2sumtkpterr[ic] > 0)
      calo2sumtkpterr[ic] = std::sqrt(calo2sumtkpterr[ic]);
  }
}

void PFAlgo3::unlinkedtk_algo(Region &r, const std::vector<int> &tk2calo) const {
  // in the meantime, promote unlinked low pt tracks to hadrons
  for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
    auto &tk = r.track[itk];
    if (tk2calo[itk] != -1 || tk.muonLink || tk.used || !tk.quality(l1tpf_impl::InputTrack::PFLOOSE))
      continue;
    float maxPt = tk.quality(l1tpf_impl::InputTrack::PFTIGHT) ? tightTrackMaxInvisiblePt_ : maxInvisiblePt_;
    if (tk.floatPt() < maxPt) {
      if (debug_)
        dbgPrintf("PFAlgo3 \t track %3d (pt %7.2f) not matched to calo, kept as charged hadron\n", itk, tk.floatPt());
      auto &p = addTrackToPF(r, tk);
      p.hwStatus = GoodTK_NoCalo;
      tk.used = true;
    } else {
      if (debug_)
        dbgPrintf("PFAlgo3 \t track %3d (pt %7.2f) not matched to calo, dropped\n", itk, tk.floatPt());
      //discardTrack(r, tk, BadTK_NoCalo); // log this as discarded, for debugging
    }
  }
}

void PFAlgo3::calo_relink(Region &r,
                          const std::vector<int> &calo2ntk,
                          const std::vector<float> &calo2sumtkpt,
                          const std::vector<float> &calo2sumtkpterr) const {
  /// OPTIONAL STEP: try to recover split hadron showers (v1.0):
  //     take hadrons that are not track matched, close by a hadron which has an excess of track pt vs calo pt
  //     add this pt to the calo pt of the other cluster
  //     off by default, as it seems to not do much in jets even if it helps remove tails in single-pion events
  std::vector<float> addtopt(r.calo.size(), 0);
  for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
    auto &calo = r.calo[ic];
    if (calo2ntk[ic] != 0 || calo.used || r.globalAbsEta(calo.floatEta()) > 2.5)
      continue;
    int i2best = -1;
    float drbest = caloReLinkDr_;
    for (int ic2 = 0; ic2 < nc; ++ic2) {
      const auto &calo2 = r.calo[ic2];
      if (calo2ntk[ic2] == 0 || calo2.used || r.globalAbsEta(calo2.floatEta()) > 2.5)
        continue;
      float dr = floatDR(calo, calo2);
      //// uncomment below for more verbose debugging
      //if (debug_ && dr < 0.5) dbgPrintf("PFAlgo3 \t calo  %3d (pt %7.2f) with no tracks is at dr %.3f from calo %3d with pt %7.2f (sum tk pt %7.2f), track excess %7.2f +- %7.2f\n", ic, calo.floatPt(), dr, ic2, calo2.floatPt(), calo2sumtkpt[ic2], calo2sumtkpt[ic2] - calo2.floatPt(), useTrackCaloSigma_ ? calo2sumtkpterr[ic2] : calo2.floatPtErr());
      if (dr < drbest) {
        float ptdiff =
            calo2sumtkpt[ic2] - calo2.floatPt() + (useTrackCaloSigma_ ? calo2sumtkpterr[ic2] : calo2.floatPtErr());
        if (ptdiff >= caloReLinkThreshold_ * calo.floatPt()) {
          i2best = ic2;
          drbest = dr;
        }
      }
    }
    if (i2best != -1) {
      const auto &calo2 = r.calo[i2best];
      if (debug_)
        dbgPrintf(
            "PFAlgo3 \t calo  %3d (pt %7.2f) with no tracks matched within dr %.3f with calo %3d with pt %7.2f (sum tk "
            "pt %7.2f), track excess %7.2f +- %7.2f\n",
            ic,
            calo.floatPt(),
            drbest,
            i2best,
            calo2.floatPt(),
            calo2sumtkpt[i2best],
            calo2sumtkpt[i2best] - calo2.floatPt(),
            useTrackCaloSigma_ ? calo2sumtkpterr[i2best] : calo2.floatPtErr());
      calo.used = true;
      addtopt[i2best] += calo.floatPt();
    }
  }
  // we do this at the end, so that the above loop is parallelizable
  for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
    if (addtopt[ic]) {
      auto &calo = r.calo[ic];
      if (debug_)
        dbgPrintf("PFAlgo3 \t calo  %3d (pt %7.2f, sum tk pt %7.2f) is increased to pt %7.2f after merging\n",
                  ic,
                  calo.floatPt(),
                  calo2sumtkpt[ic],
                  calo.floatPt() + addtopt[ic]);
      calo.setFloatPt(calo.floatPt() + addtopt[ic]);
    }
  }
}

void PFAlgo3::linkedcalo_algo(Region &r,
                              const std::vector<int> &calo2ntk,
                              const std::vector<float> &calo2sumtkpt,
                              const std::vector<float> &calo2sumtkpterr,
                              std::vector<float> &calo2alpha) const {
  /// ------------- next step (needs the previous) ----------------
  // process matched calo clusters, compare energy to sum track pt
  for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
    auto &calo = r.calo[ic];
    if (calo2ntk[ic] == 0 || calo.used)
      continue;
    float ptdiff = calo.floatPt() - calo2sumtkpt[ic];
    float pterr = useTrackCaloSigma_ ? calo2sumtkpterr[ic] : calo.floatPtErr();
    if (debug_)
      dbgPrintf(
          "PFAlgo3 \t calo  %3d (pt %7.2f +- %7.2f, empt %7.2f) has %2d tracks (sumpt %7.2f, sumpterr %7.2f), ptdif "
          "%7.2f +- %7.2f\n",
          ic,
          calo.floatPt(),
          calo.floatPtErr(),
          calo.floatEmPt(),
          calo2ntk[ic],
          calo2sumtkpt[ic],
          calo2sumtkpterr[ic],
          ptdiff,
          pterr);
    if (ptdiff > +ptMatchHigh_ * pterr) {
      if (ecalPriority_) {
        if (calo.floatEmPt() > 1) {
          float emptdiff = std::min(ptdiff, calo.floatEmPt());
          if (debug_)
            dbgPrintf(
                "PFAlgo3 \t calo  %3d (pt %7.2f, empt %7.2f)    ---> make photon with pt %7.2f, reduce ptdiff to %7.2f "
                "+- %7.2f\n",
                ic,
                calo.floatPt(),
                calo.floatEmPt(),
                emptdiff,
                ptdiff - emptdiff,
                pterr);
          auto &p = addCaloToPF(r, calo);
          p.setFloatPt(emptdiff);
          p.hwId = l1t::PFCandidate::Photon;
          ptdiff -= emptdiff;
        }
        if (ptdiff > 2) {
          if (debug_)
            dbgPrintf("PFAlgo3 \t calo  %3d (pt %7.2f, empt %7.2f)    ---> make also neutral hadron with pt %7.2f\n",
                      ic,
                      calo.floatPt(),
                      calo.floatEmPt(),
                      ptdiff);
          auto &p = addCaloToPF(r, calo);
          p.setFloatPt(ptdiff);
          p.hwId = l1t::PFCandidate::NeutralHadron;
        }
      } else {
        if (debug_)
          dbgPrintf("PFAlgo3 \t calo  %3d (pt %7.2f)    ---> promoted to neutral with pt %7.2f\n",
                    ic,
                    calo.floatPt(),
                    ptdiff);
        auto &p = addCaloToPF(r, calo);
        p.setFloatPt(ptdiff);
        calo.hwFlags = 0;
      }
    } else if (ptdiff > -ptMatchLow_ * pterr) {
      // nothing to do (weighted average happens when we process the tracks)
      calo.hwFlags = 1;
      if (debug_)
        dbgPrintf(
            "PFAlgo3 \t calo  %3d (pt %7.2f)    ---> to be deleted, will use tracks instead\n", ic, calo.floatPt());
      //discardCalo(r, calo, 0); // log this as discarded, for debugging
    } else {
      // tracks overshoot, rescale to tracks to calo
      calo2alpha[ic] = rescaleTracks_ ? calo.floatPt() / calo2sumtkpt[ic] : 1.0;
      calo.hwFlags = 2;
      if (debug_ && rescaleTracks_)
        dbgPrintf("PFAlgo3 \t calo  %3d (pt %7.2f)    ---> tracks overshoot and will be scaled down by %.4f\n",
                  ic,
                  calo.floatPt(),
                  calo2alpha[ic]);
      if (debug_ && !rescaleTracks_)
        dbgPrintf("PFAlgo3 \t calo  %3d (pt %7.2f)    ---> tracks overshoot by %.4f\n",
                  ic,
                  calo.floatPt(),
                  calo2sumtkpt[ic] / calo.floatPt());
    }
    calo.used = true;
  }
}

void PFAlgo3::linkedtk_algo(Region &r,
                            const std::vector<int> &tk2calo,
                            const std::vector<int> &calo2ntk,
                            const std::vector<float> &calo2alpha) const {
  // process matched tracks, if necessary rescale or average
  for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
    auto &tk = r.track[itk];
    if (tk2calo[itk] == -1 || tk.muonLink || tk.used)
      continue;
    auto &p = addTrackToPF(r, tk);
    tk.used = true;
    const auto &calo = r.calo[tk2calo[itk]];
    p.cluster.src = calo.src;
    if (calo.hwFlags == 1) {
      // can do weighted average if there's just one track
      if (calo2ntk[tk2calo[itk]] == 1 && caloTrkWeightedAverage_) {
        p.hwStatus = GoodTK_Calo_TkPt;
        float ptavg = tk.floatPt();
        if (tk.floatPtErr() > 0) {
          float wcalo = 1.0 / std::pow(tk.floatCaloPtErr(), 2);
          float wtk = 1.0 / std::pow(tk.floatPtErr(), 2);
          ptavg = (calo.floatPt() * wcalo + tk.floatPt() * wtk) / (wcalo + wtk);
          p.hwStatus = GoodTK_Calo_TkCaloPt;
        }
        p.setFloatPt(ptavg);
        if (debug_)
          dbgPrintf(
              "PFAlgo3 \t track %3d (pt %7.2f +- %7.2f) combined with calo %3d (pt %7.2f +- %7.2f (from tk) yielding "
              "candidate of pt %7.2f\n",
              itk,
              tk.floatPt(),
              tk.floatPtErr(),
              tk2calo[itk],
              calo.floatPt(),
              tk.floatCaloPtErr(),
              ptavg);
      } else {
        p.hwStatus = GoodTK_Calo_TkPt;
        if (debug_)
          dbgPrintf("PFAlgo3 \t track %3d (pt %7.2f) linked to calo %3d promoted to charged hadron\n",
                    itk,
                    tk.floatPt(),
                    tk2calo[itk]);
      }
    } else if (calo.hwFlags == 2) {
      // must rescale
      p.setFloatPt(tk.floatPt() * calo2alpha[tk2calo[itk]]);
      p.hwStatus = GoodTk_Calo_CaloPt;
      if (debug_)
        dbgPrintf(
            "PFAlgo3 \t track %3d (pt %7.2f, stubs %2d chi2 %7.1f) linked to calo %3d promoted to charged hadron with "
            "pt %7.2f after maybe rescaling\n",
            itk,
            tk.floatPt(),
            int(tk.hwStubs),
            tk.hwChi2 * 0.1f,
            tk2calo[itk],
            p.floatPt());
    }
  }
}

void PFAlgo3::unlinkedcalo_algo(Region &r) const {
  // process unmatched calo clusters
  for (int ic = 0, nc = r.calo.size(); ic < nc; ++ic) {
    if (!r.calo[ic].used) {
      addCaloToPF(r, r.calo[ic]);
      if (debug_)
        dbgPrintf("PFAlgo3 \t calo  %3d (pt %7.2f) not linked, promoted to neutral\n", ic, r.calo[ic].floatPt());
    }
  }
}

void PFAlgo3::save_muons(Region &r, const std::vector<int> &tk2mu) const {
  // finally do muons
  for (int itk = 0, ntk = r.track.size(); itk < ntk; ++itk) {
    if (r.track[itk].muonLink) {
      auto &p = addTrackToPF(r, r.track[itk]);
      p.muonsrc = r.muon[tk2mu[itk]].src;
      if (debug_)
        dbgPrintf("PFAlgo3 \t track %3d (pt %7.2f) promoted to muon.\n", itk, r.track[itk].floatPt());
    }
  }
}
