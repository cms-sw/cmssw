#include "L1Trigger/Phase2L1ParticleFlow/interface/pf/pfalgo_common_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <cmath>
#include <cstdio>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#endif

l1ct::PFAlgoEmulatorBase::~PFAlgoEmulatorBase() {}

void l1ct::PFAlgoEmulatorBase::loadPtErrBins(
    unsigned int nbins, const float absetas[], const float scales[], const float offs[], bool verbose) {
  ptErrBins_.resize(nbins);
  for (unsigned int i = 0; i < nbins; ++i) {
    ptErrBins_[i].abseta = Scales::makeGlbEta(absetas[i]);
    ptErrBins_[i].scale = scales[i];
    ptErrBins_[i].offs = offs[i];

    if (verbose || debug_)
      dbgPrintf("loadPtErrBins: #%d: abseta %5.3f -> %8d, scale %7.4f -> %7.4f, offs %7.3f -> %7.4f\n",
                i,
                absetas[i],
                ptErrBins_[i].abseta.to_int(),
                scales[i],
                ptErrBins_[i].scale.to_float(),
                offs[i],
                ptErrBins_[i].offs.to_float());
  }
}

#ifdef CMSSW_GIT_HASH
void l1ct::PFAlgoEmulatorBase::loadPtErrBins(const edm::ParameterSet &iConfig) {
  const edm::ParameterSet &resol = iConfig.getParameter<edm::ParameterSet>("caloResolution");
  std::vector<float> absetas, scales, offs;
  for (auto &v : resol.getParameter<std::vector<double>>("etaBins"))
    absetas.push_back(v);
  for (auto &v : resol.getParameter<std::vector<double>>("scale"))
    scales.push_back(v);
  for (auto &v : resol.getParameter<std::vector<double>>("offset"))
    offs.push_back(v);
  loadPtErrBins(absetas.size(), &absetas[0], &scales[0], &offs[0]);
}

#endif

l1ct::pt_t l1ct::PFAlgoEmulatorBase::ptErr_ref(const l1ct::PFRegionEmu &region, const l1ct::TkObjEmu &track) const {
  glbeta_t abseta = region.hwGlbEta(track.hwEta);
  if (abseta < 0)
    abseta = -abseta;

  ptErrScale_t scale = 0.3125;
  ptErrOffs_t offs = 7.0;
  for (const auto &bin : ptErrBins_) {
    if (abseta < bin.abseta) {
      scale = bin.scale;
      offs = bin.offs;
      break;
    }
  }

  pt_t ptErr = track.hwPt * scale + offs;
  if (ptErr > track.hwPt)
    ptErr = track.hwPt;
  return ptErr;
}

void l1ct::PFAlgoEmulatorBase::pfalgo_mu_ref(const PFInputRegion &in, OutputRegion &out, std::vector<int> &iMu) const {
  // init
  unsigned int nTRACK = std::min<unsigned>(nTRACK_, in.track.size());
  unsigned int nMU = std::min<unsigned>(nMU_, in.muon.size());
  out.pfmuon.resize(nMU);
  iMu.resize(nTRACK);
  for (unsigned int ipf = 0; ipf < nMU; ++ipf)
    out.pfmuon[ipf].clear();
  for (unsigned int it = 0; it < nTRACK; ++it)
    iMu[it] = -1;

  // for each muon, find the closest track
  for (unsigned int im = 0; im < nMU; ++im) {
    if (in.muon[im].hwPt > 0) {
      int ibest = -1;
      pt_t dptmin = in.muon[im].hwPt >> 1;
      for (unsigned int it = 0; it < nTRACK; ++it) {
        if (!in.track[it].isPFLoose())
          continue;
        unsigned int dr = dr2_int(in.muon[im].hwEta, in.muon[im].hwPhi, in.track[it].hwEta, in.track[it].hwPhi);
        //dbgPrintf("deltaR2(mu %d float pt %5.1f, tk %2d float pt %5.1f) = int %d  (float deltaR = %.3f); int cut at %d\n", im, 0.25*int(in.muon[im].hwPt), it, 0.25*int(in.track[it].hwPt), dr, std::sqrt(float(dr))/229.2, dR2MAX_TK_MU_);
        if (dr < dR2MAX_TK_MU_) {
          dpt_t dpt = (dpt_t(in.track[it].hwPt) - dpt_t(in.muon[im].hwPt));
          pt_t absdpt = dpt >= 0 ? pt_t(dpt) : pt_t(-dpt);
          if (absdpt < dptmin) {
            dptmin = absdpt;
            ibest = it;
          }
        }
      }
      if (ibest != -1) {
        iMu[ibest] = im;
        fillPFCand(in.track[ibest], out.pfmuon[im], /*isMu=*/true, /*isEle=*/false);
        // extra emulator info
        out.pfmuon[im].srcMu = in.muon[im].src;
        if (debug_)
          dbgPrintf("FW  \t muon %3d linked to track %3d \n", im, ibest);
      } else {
        if (debug_)
          dbgPrintf("FW  \t muon %3d not linked to any track\n", im);
      }
    }
  }
}

void l1ct::PFAlgoEmulatorBase::fillPFCand(const TkObjEmu &track, PFChargedObjEmu &pf, bool isMu, bool isEle) const {
  assert(!(isEle && isMu));
  pf.hwPt = track.hwPt;
  pf.hwEta = track.hwEta;
  pf.hwPhi = track.hwPhi;
  pf.hwDEta = track.hwDEta;
  pf.hwDPhi = track.hwDPhi;
  pf.hwZ0 = track.hwZ0;
  pf.hwDxy = track.hwDxy;
  pf.hwTkQuality = track.hwQuality;
  if (isMu) {
    pf.hwId = ParticleID::mkMuon(track.hwCharge);
  } else if (isEle) {
    pf.hwId = ParticleID::mkElectron(track.hwCharge);
  } else {
    pf.hwId = ParticleID::mkChHad(track.hwCharge);
  }
  // extra emulator information
  pf.srcTrack = track.src;
}

void l1ct::PFAlgoEmulatorBase::fillPFCand(const HadCaloObjEmu &calo, PFNeutralObjEmu &pf, bool isPhoton) const {
  pf.hwPt = calo.hwPt;
  pf.hwEta = calo.hwEta;
  pf.hwPhi = calo.hwPhi;
  pf.hwId = isPhoton ? ParticleID::PHOTON : ParticleID::HADZERO;
  pf.hwEmPt = calo.hwEmPt;  // FIXME
  pf.hwEmID = calo.hwEmID;
  pf.hwPUID = 0;
  // extra emulator information
  pf.srcCluster = calo.src;
}

void l1ct::PFAlgoEmulatorBase::fillPFCand(const EmCaloObjEmu &calo, PFNeutralObjEmu &pf, bool isPhoton) const {
  pf.hwPt = calo.hwPt;
  pf.hwEta = calo.hwEta;
  pf.hwPhi = calo.hwPhi;
  pf.hwId = isPhoton ? ParticleID::PHOTON : ParticleID::HADZERO;
  pf.hwEmPt = calo.hwPt;
  pf.hwEmID = calo.hwEmID;
  pf.hwPUID = 0;
  // more emulator info
  pf.srcCluster = calo.src;
}
