#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pftkegalgo_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <iostream>
#include <bitset>
#include <vector>

#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"


using namespace l1ct;

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::PFTkEGAlgoEmuConfig::PFTkEGAlgoEmuConfig(const edm::ParameterSet &pset)
    : nTRACK(pset.getParameter<uint32_t>("nTRACK")),
      nTRACK_EGIN(pset.getParameter<uint32_t>("nTRACK_EGIN")),
      nEMCALO_EGIN(pset.getParameter<uint32_t>("nEMCALO_EGIN")),
      nEM_EGOUT(pset.getParameter<uint32_t>("nEM_EGOUT")),
      filterHwQuality(pset.getParameter<bool>("filterHwQuality")),
      doBremRecovery(pset.getParameter<bool>("doBremRecovery")),
      writeBeforeBremRecovery(pset.getParameter<bool>("writeBeforeBremRecovery")),
      caloHwQual(pset.getParameter<int>("caloHwQual")),
      doEndcapHwQual(pset.getParameter<bool>("doEndcapHwQual")),
      emClusterPtMin(pset.getParameter<double>("caloEtMin")),
      dEtaMaxBrem(pset.getParameter<double>("dEtaMaxBrem")),
      dPhiMaxBrem(pset.getParameter<double>("dPhiMaxBrem")),
      absEtaBoundaries(pset.getParameter<std::vector<double>>("absEtaBoundaries")),
      dEtaValues(pset.getParameter<std::vector<double>>("dEtaValues")),
      dPhiValues(pset.getParameter<std::vector<double>>("dPhiValues")),
      trkQualityPtMin(pset.getParameter<double>("trkQualityPtMin")),
      doCompositeTkEle(pset.getParameter<bool>("doCompositeTkEle")),
      nCOMPCAND_PER_CLUSTER(pset.getParameter<uint32_t>("nCOMPCAND_PER_CLUSTER")),
      writeEgSta(pset.getParameter<bool>("writeEGSta")),
      tkIsoParams_tkEle(pset.getParameter<edm::ParameterSet>("tkIsoParametersTkEle")),
      tkIsoParams_tkEm(pset.getParameter<edm::ParameterSet>("tkIsoParametersTkEm")),
      pfIsoParams_tkEle(pset.getParameter<edm::ParameterSet>("pfIsoParametersTkEle")),
      pfIsoParams_tkEm(pset.getParameter<edm::ParameterSet>("pfIsoParametersTkEm")),
      doTkIso(pset.getParameter<bool>("doTkIso")),
      doPfIso(pset.getParameter<bool>("doPfIso")),
      hwIsoTypeTkEle(static_cast<EGIsoEleObjEmu::IsoType>(pset.getParameter<uint32_t>("hwIsoTypeTkEle"))),
      hwIsoTypeTkEm(static_cast<EGIsoObjEmu::IsoType>(pset.getParameter<uint32_t>("hwIsoTypeTkEm"))),
      compIDparams(pset.getParameter<edm::ParameterSet>("compositeParametersTkEle")),
      debug(pset.getUntrackedParameter<uint32_t>("debug", 0)) {}

l1ct::PFTkEGAlgoEmuConfig::IsoParameters::IsoParameters(const edm::ParameterSet &pset)
    : IsoParameters(pset.getParameter<double>("tkQualityPtMin"),
                    pset.getParameter<double>("dZ"),
                    pset.getParameter<double>("dRMin"),
                    pset.getParameter<double>("dRMax")) {}

l1ct::PFTkEGAlgoEmuConfig::CompIDParameters::CompIDParameters(const edm::ParameterSet &pset)
    : CompIDParameters(pset.getParameter<double>("hoeMin"),
                       pset.getParameter<double>("hoeMax"),
                       pset.getParameter<double>("tkptMin"),
                       pset.getParameter<double>("tkptMax"),
                       pset.getParameter<double>("srrtotMin"),
                       pset.getParameter<double>("srrtotMax"),
                       pset.getParameter<double>("detaMin"),
                       pset.getParameter<double>("detaMax"),
                       pset.getParameter<double>("dptMin"),
                       pset.getParameter<double>("dptMax"),
                       pset.getParameter<double>("meanzMin"),
                       pset.getParameter<double>("meanzMax"),
                       pset.getParameter<double>("dphiMin"),
                       pset.getParameter<double>("dphiMax"),
                       pset.getParameter<double>("tkchi2Min"),
                       pset.getParameter<double>("tkchi2Max"),
                       pset.getParameter<double>("tkz0Min"),
                       pset.getParameter<double>("tkz0Max"),
                       pset.getParameter<double>("tknstubsMin"),
                       pset.getParameter<double>("tknstubsMax"),
                       pset.getParameter<double>("BDTcut_wp97p5"),
                       pset.getParameter<double>("BDTcut_wp95p0")) {}

#endif

PFTkEGAlgoEmulator::PFTkEGAlgoEmulator(const PFTkEGAlgoEmuConfig &config) : cfg(config), 
composite_bdt_(nullptr), 
debug_(cfg.debug) {
  if(cfg.doCompositeTkEle) {
    //FIXME: make the name of the file configurable
#ifdef CMSSW_GIT_HASH
	  auto resolvedFileName = edm::FileInPath("L1Trigger/Phase2L1ParticleFlow/data/compositeID.json").fullPath();
#else
          auto resolvedFileName = "compositeID.json";
#endif
    std::cout<<resolvedFileName<<std::endl;
	  composite_bdt_ = new conifer::BDT<ap_fixed<21,12,AP_RND_CONV,AP_SAT>,ap_fixed<12,3,AP_RND_CONV,AP_SAT>,0> (resolvedFileName);
    std::cout<<"declared bdt"<<std::endl;
  }
}

void PFTkEGAlgoEmulator::toFirmware(const PFInputRegion &in,
                                    PFRegion &region,
                                    EmCaloObj emcalo[/*nCALO*/],
                                    TkObj track[/*nTRACK*/]) const {
  region = in.region;
  l1ct::toFirmware(in.track, cfg.nTRACK_EGIN, track);
  l1ct::toFirmware(in.emcalo, cfg.nEMCALO_EGIN, emcalo);
  if (debug_ > 0)
    dbgCout() << "# of inpput tracks: " << in.track.size() << " (max: " << cfg.nTRACK_EGIN << ")"
              << " emcalo: " << in.emcalo.size() << "(" << cfg.nEMCALO_EGIN << ")" << std::endl;
}

void PFTkEGAlgoEmulator::toFirmware(const OutputRegion &out, EGIsoObj out_egphs[], EGIsoEleObj out_egeles[]) const {
  l1ct::toFirmware(out.egphoton, cfg.nEM_EGOUT, out_egphs);
  l1ct::toFirmware(out.egelectron, cfg.nEM_EGOUT, out_egeles);
  if (debug_ > 0)
    dbgCout() << "# output photons: " << out.egphoton.size() << " electrons: " << out.egelectron.size() << std::endl;
}

void PFTkEGAlgoEmulator::toFirmware(
    const PFInputRegion &in, const l1ct::PVObjEmu &pvin, PFRegion &region, TkObj track[/*nTRACK*/], PVObj &pv) const {
  region = in.region;
  l1ct::toFirmware(in.track, cfg.nTRACK, track);
  pv = pvin;
  if (debug_ > 0)
    dbgCout() << "# of inpput tracks: " << in.track.size() << " (max: " << cfg.nTRACK << ")" << std::endl;
}

float PFTkEGAlgoEmulator::deltaPhi(float phi1, float phi2) const {
  // reduce to [-pi,pi]
  float x = phi1 - phi2;
  float o2pi = 1. / (2. * M_PI);
  if (std::abs(x) <= float(M_PI))
    return x;
  float n = std::round(x * o2pi);
  return x - n * float(2. * M_PI);
}

void PFTkEGAlgoEmulator::link_emCalo2emCalo(const std::vector<EmCaloObjEmu> &emcalo,
                                            std::vector<int> &emCalo2emCalo) const {
  // NOTE: we assume the input to be sorted!!!
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];
    if (emCalo2emCalo[ic] != -1)
      continue;

    for (int jc = ic + 1; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] != -1)
        continue;

      auto &otherCalo = emcalo[jc];

      if (fabs(otherCalo.floatEta() - calo.floatEta()) < cfg.dEtaMaxBrem &&
          fabs(deltaPhi(otherCalo.floatPhi(), calo.floatPhi())) < cfg.dPhiMaxBrem) {
        emCalo2emCalo[jc] = ic;
      }
    }
  }
}


void PFTkEGAlgoEmulator::link_emCalo2tk_elliptic(const PFRegionEmu &r,
                                                 const std::vector<EmCaloObjEmu> &emcalo,
                                                 const std::vector<TkObjEmu> &track,
                                                 std::vector<int> &emCalo2tk) const {
  unsigned int nTrackMax = std::min<unsigned>(track.size(), cfg.nTRACK_EGIN);
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];

    float dPtMin = 999;
    for (unsigned int itk = 0; itk < nTrackMax; ++itk) {
      const auto &tk = track[itk];
      if (tk.floatPt() < cfg.trkQualityPtMin)
        continue;

      float d_phi = deltaPhi(tk.floatPhi(), calo.floatPhi());
      float d_eta = tk.floatEta() - calo.floatEta();  // We only use it squared

      auto eta_index =
          std::distance(cfg.absEtaBoundaries.begin(),
                        std::lower_bound(
                            cfg.absEtaBoundaries.begin(), cfg.absEtaBoundaries.end(), abs(r.floatGlbEta(calo.hwEta)))) -
          1;

      float dEtaMax = cfg.dEtaValues[eta_index];
      float dPhiMax = cfg.dPhiValues[eta_index];

      if ((((d_phi / dPhiMax) * (d_phi / dPhiMax)) + ((d_eta / dEtaMax) * (d_eta / dEtaMax))) < 1.) {
        // NOTE: for now we implement only best pt match. This is NOT what is done in the L1TkElectronTrackProducer
        if (fabs(tk.floatPt() - calo.floatPt()) < dPtMin) {
          emCalo2tk[ic] = itk;
          dPtMin = fabs(tk.floatPt() - calo.floatPt());
        }
      }
    }
  }
}


void PFTkEGAlgoEmulator::link_emCalo2tk_composite(const PFRegionEmu &r,
                                        const std::vector<EmCaloObjEmu> &emcalo,
                                        const std::vector<TkObjEmu> &track,
                                        std::vector<int> &emCalo2tk, 
                                        std::vector<float> &emCaloTkBdtScore) const {
  unsigned int nTrackMax = std::min<unsigned>(track.size(), cfg.nTRACK_EGIN);
  std::cout<<"doing loose dR matching"<<std::endl;
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    std::cout<<"cluster "<<ic<<std::endl;
    auto &calo = emcalo[ic];

    std::vector<CompositeCandidate> candidates;

    for (unsigned int itk = 0; itk < nTrackMax; ++itk) {
      std::cout<<"track "<<itk<<std::endl;
      const auto &tk = track[itk];
      if (tk.floatPt() <= cfg.trkQualityPtMin)
        continue;

      float d_phi = deltaPhi(tk.floatPhi(), calo.floatPhi());
      float d_eta = tk.floatEta() - calo.floatEta();  // We only use it squared
      float dR = sqrt((d_phi * d_phi ) + (d_eta * d_eta ));

      if (dR<0.2){
          // Only store indices, dR and dpT for now. The other quantities are computed only for the best nCandPerCluster.
          CompositeCandidate cand;
          cand.cluster_idx = ic;
          cand.track_idx = itk;
          cand.dpt = fabs(tk.floatPt() - calo.floatPt());
          candidates.push_back(cand);
      }
    }
    std::cout << "Constructed candidates, now sorting" << std::endl;
    // FIXME: find best sort criteria, for now we use dpt
    std::sort(candidates.begin(), candidates.end(), 
              [](const CompositeCandidate & a, const CompositeCandidate & b) -> bool
                { return a.dpt < b.dpt; });
    unsigned int nCandPerCluster = std::min<unsigned int>(candidates.size(), cfg.nCOMPCAND_PER_CLUSTER);
    std::cout << "# composite candidates: " << nCandPerCluster << std::endl;
    if(nCandPerCluster == 0) continue;

    float bdtWP_MVA = cfg.compIDparams.BDTcut_wp97p5;
    float bdtWP_XGB = 1. / (1. + std::sqrt((1. - bdtWP_MVA) / (1. + bdtWP_MVA))); // Convert WP value from ROOT.TMVA to XGboost
    float maxScore = -999;
    int ibest = -1;
    for(unsigned int icand = 0; icand < nCandPerCluster; icand++) {
      auto &cand = candidates[icand];
      std::vector<EmCaloObjEmu> emcalo_sel = emcalo;
      float score = compute_composite_score(cand, emcalo_sel, track, cfg.compIDparams);
      if(score > maxScore) {
      // if((score > bdtWP_XGB) && (score > maxScore)) {
        maxScore = score;
        ibest = icand;
      }
    }
    if(ibest != -1) {
      emCalo2tk[ic] = candidates[ibest].track_idx;
      emCaloTkBdtScore[ic] = maxScore;
    }
  }
}


float PFTkEGAlgoEmulator::compute_composite_score(CompositeCandidate &cand,
                                                  const std::vector<EmCaloObjEmu> &emcalo,
                                                  const std::vector<TkObjEmu> &track,
                                                  const PFTkEGAlgoEmuConfig::CompIDParameters &params) const {
  // Get the cluster/track objects that form the composite candidate
  const auto &calo = emcalo[cand.cluster_idx];
  const auto &tk = track[cand.track_idx];

  // Call and normalize input feature values, then cast to ap_fixed.
  // Note that for some features (e.g. track pT) we call the floating point representation, but that's already quantized!
  // Several other features, such as chi2 or most cluster features, are not quantized before casting them to ap_fixed.
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> hoe = calo.floatHoe();
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> tkpt = tk.floatPt();
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> srrtot = calo.floatSrrTot();
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> deta = tk.floatEta() - calo.floatEta();
  // FIXME: do we really need dpt to be a ratio?
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> dpt = (tk.floatPt()/calo.floatPt());
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> meanz = calo.floatMeanZ();
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> dphi = deltaPhi(tk.floatPhi(), calo.floatPhi());
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> chi2 = tk.floatChi2();
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> tkz0 = tk.floatZ0();
  ap_fixed<21,12,AP_RND_CONV,AP_SAT> nstubs = tk.hwStubs;
  
  // Run BDT inference
  std::vector<ap_fixed<21,12,AP_RND_CONV,AP_SAT>> inputs = { hoe, tkpt, srrtot, deta, dpt, meanz, dphi, chi2, tkz0, nstubs } ;
  std::vector<ap_fixed<12,3,AP_RND_CONV,AP_SAT>> bdt_score = composite_bdt_->decision_function(inputs);

  float bdt_score_CON = bdt_score[0];
  float bdt_score_XGB = 1/(1+exp(-bdt_score_CON)); // Map Conifer score to XGboost score. (same as scipy.expit)

  // std::cout<<"BDT score of composite candidate = "<<bdt_score_XGB<<std::endl;
  return bdt_score_XGB;
}


void PFTkEGAlgoEmulator::sel_emCalo(unsigned int nmax_sel,
                                    const std::vector<EmCaloObjEmu> &emcalo,
                                    std::vector<EmCaloObjEmu> &emcalo_sel) const {
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    const auto &calo = emcalo[ic];
    if ((calo.hwPt == 0) || (cfg.filterHwQuality && calo.hwEmID != cfg.caloHwQual) ||
        (calo.floatPt() < cfg.emClusterPtMin))
      continue;
    emcalo_sel.push_back(calo);
    if (emcalo_sel.size() >= nmax_sel)
      break;
  }
}

void PFTkEGAlgoEmulator::run(const PFInputRegion &in, OutputRegion &out) const {
  if (debug_ > 1) {
    for (int ic = 0, nc = in.emcalo.size(); ic < nc; ++ic) {
      const auto &calo = in.emcalo[ic];
      if (calo.hwPt > 0)
        dbgCout() << "[REF] IN calo[" << ic << "] pt: " << calo.hwPt << " eta: " << calo.hwEta
                  << " (glb eta: " << in.region.floatGlbEta(calo.hwEta) << ") phi: " << calo.hwPhi
                  << "(glb phi: " << in.region.floatGlbPhi(calo.hwPhi) << ") qual: " << std::bitset<4>(calo.hwEmID)
                  << std::endl;
    }
  }
  std::cout<<"running"<<std::endl;
  // FIXME: can be removed in the endcap since now running with the "interceptor".
  // Might still be needed in barrel
  // filter and select first N elements of input clusters
  std::vector<EmCaloObjEmu> emcalo_sel;
  sel_emCalo(cfg.nEMCALO_EGIN, in.emcalo, emcalo_sel);

  std::vector<int> emCalo2emCalo(emcalo_sel.size(), -1);
  if (cfg.doBremRecovery)
    link_emCalo2emCalo(emcalo_sel, emCalo2emCalo);

  std::vector<int> emCalo2tk(emcalo_sel.size(), -1);
  std::vector<float> emCaloTkBdtScore(emcalo_sel.size(), -999);
  std::cout<<"about to start matching"<<std::endl;

  if(cfg.doCompositeTkEle) {
    link_emCalo2tk_composite(in.region, emcalo_sel, in.track, emCalo2tk, emCaloTkBdtScore);
  } else {
    link_emCalo2tk_elliptic(in.region, emcalo_sel, in.track, emCalo2tk);
  }
  
  out.egsta.clear();
  std::vector<EGIsoObjEmu> egobjs;
  std::vector<EGIsoEleObjEmu> egeleobjs;
  eg_algo(in.region, emcalo_sel, in.track, emCalo2emCalo, emCalo2tk, emCaloTkBdtScore, out.egsta, egobjs, egeleobjs);

  unsigned int nEGOut = std::min<unsigned>(cfg.nEM_EGOUT, egobjs.size());
  unsigned int nEGEleOut = std::min<unsigned>(cfg.nEM_EGOUT, egeleobjs.size());

  // init output containers
  out.egphoton.clear();
  out.egelectron.clear();
  ptsort_ref(egobjs.size(), nEGOut, egobjs, out.egphoton);
  ptsort_ref(egeleobjs.size(), nEGEleOut, egeleobjs, out.egelectron);
}

void PFTkEGAlgoEmulator::eg_algo(const PFRegionEmu &region,
                                 const std::vector<EmCaloObjEmu> &emcalo,
                                 const std::vector<TkObjEmu> &track,
                                 const std::vector<int> &emCalo2emCalo,
                                 const std::vector<int> &emCalo2tk,
                                 const std::vector<float> &emCaloTkBdtScore,
                                 std::vector<EGObjEmu> &egstas,
                                 std::vector<EGIsoObjEmu> &egobjs,
                                 std::vector<EGIsoEleObjEmu> &egeleobjs) const {
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];

    // discard immediately EG objects that would not fall in the fiducial eta-phi region
    if (!region.isFiducial(calo))
      continue;

    if (debug_ > 3)
      dbgCout() << "[REF] SEL emcalo with pt: " << calo.hwPt << " qual: " << calo.hwEmID << " eta: " << calo.hwEta
                << " phi " << calo.hwPhi << std::endl;

    int itk = emCalo2tk[ic];
    float bdt = emCaloTkBdtScore[ic];

    // check if brem recovery is on
    if (!cfg.doBremRecovery || cfg.writeBeforeBremRecovery) {
      // 1. create EG objects before brem recovery
      unsigned int egQual = calo.hwEmID;
      // If we write both objects with and without brem-recovery
      // bit 3 is used for the brem-recovery bit: if set = no recovery
      // (for consistency with the barrel hwQual where by default the brem recovery is done upstream)
      if (cfg.writeBeforeBremRecovery && cfg.doBremRecovery) {
        egQual = calo.hwEmID | 0x8;
      }

      addEgObjsToPF(egstas, egobjs, egeleobjs, emcalo, track, ic, egQual, calo.hwPt, itk, bdt);
    }

    if (!cfg.doBremRecovery)
      continue;

    // check if the cluster has already been used in a brem reclustering
    if (emCalo2emCalo[ic] != -1)
      continue;

    pt_t ptBremReco = calo.hwPt;
    std::vector<unsigned int> components;

    for (int jc = ic; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] == ic) {
        auto &otherCalo = emcalo[jc];
        ptBremReco += otherCalo.hwPt;
        components.push_back(jc);
      }
    }

    // 2. create EG objects with brem recovery
    // NOTE: duplicating the object is suboptimal but this is done for keeping things as in TDR code...
    addEgObjsToPF(egstas, egobjs, egeleobjs, emcalo, track, ic, calo.hwEmID, ptBremReco, itk, bdt, components);
  }
}

EGObjEmu &PFTkEGAlgoEmulator::addEGStaToPF(std::vector<EGObjEmu> &egobjs,
                                           const EmCaloObjEmu &calo,
                                           const unsigned int hwQual,
                                           const pt_t ptCorr,
                                           const std::vector<unsigned int> &components) const {
  EGObjEmu egsta;
  egsta.clear();
  egsta.hwPt = ptCorr;
  egsta.hwEta = calo.hwEta;
  egsta.hwPhi = calo.hwPhi;
  egsta.hwQual = hwQual;
  egobjs.push_back(egsta);

  if (debug_ > 2)
    dbgCout() << "[REF] EGSta pt: " << egsta.hwPt << " eta: " << egsta.hwEta << " phi: " << egsta.hwPhi
              << " qual: " << std::bitset<4>(egsta.hwQual) << " packed: " << egsta.pack().to_string(16) << std::endl;

  return egobjs.back();
}

EGIsoObjEmu &PFTkEGAlgoEmulator::addEGIsoToPF(std::vector<EGIsoObjEmu> &egobjs,
                                              const EmCaloObjEmu &calo,
                                              const unsigned int hwQual,
                                              const pt_t ptCorr) const {
  EGIsoObjEmu egiso;
  egiso.clear();
  egiso.hwPt = ptCorr;
  egiso.hwEta = calo.hwEta;
  egiso.hwPhi = calo.hwPhi;
  unsigned int egHwQual = hwQual;
  if (cfg.doEndcapHwQual) {
    // 1. zero-suppress the loose EG-ID (bit 1)
    // 2. for now use the standalone tight definition (bit 0) to set the tight point for photons (bit 2)
    egHwQual = (hwQual & 0x9) | ((hwQual & 0x1) << 2);
  }
  egiso.hwQual = egHwQual;
  egiso.srcCluster = calo.src;
  egobjs.push_back(egiso);

  if (debug_ > 2)
    dbgCout() << "[REF] EGIsoObjEmu pt: " << egiso.hwPt << " eta: " << egiso.hwEta << " phi: " << egiso.hwPhi
              << " qual: " << std::bitset<4>(egiso.hwQual) << " packed: " << egiso.pack().to_string(16) << std::endl;

  return egobjs.back();
}

EGIsoEleObjEmu &PFTkEGAlgoEmulator::addEGIsoEleToPF(std::vector<EGIsoEleObjEmu> &egobjs,
                                                    const EmCaloObjEmu &calo,
                                                    const TkObjEmu &track,
                                                    const unsigned int hwQual,
                                                    const pt_t ptCorr,
                                                    const float bdtScore) const {
  EGIsoEleObjEmu egiso;
  egiso.clear();
  egiso.hwPt = ptCorr;
  egiso.hwEta = calo.hwEta;
  egiso.hwPhi = calo.hwPhi;
  unsigned int egHwQual = hwQual;
  if (cfg.doEndcapHwQual) {
    // 1. zero-suppress the loose EG-ID (bit 1)
    // 2. for now use the standalone tight definition (bit 0) to set the tight point for eles (bit 1)
    egHwQual = (hwQual & 0x9) | ((hwQual & 0x1) << 1);
  }
  egiso.hwQual = egHwQual;
  egiso.hwDEta = track.hwVtxEta() - egiso.hwEta;
  egiso.hwDPhi = abs(track.hwVtxPhi() - egiso.hwPhi);
  egiso.hwZ0 = track.hwZ0;
  egiso.hwCharge = track.hwCharge;
  egiso.srcCluster = calo.src;
  egiso.srcTrack = track.src;
  egiso.bdtScore = bdtScore;
  egobjs.push_back(egiso);

  if (debug_ > 2)
    dbgCout() << "[REF] EGIsoEleObjEmu pt: " << egiso.hwPt << " eta: " << egiso.hwEta << " phi: " << egiso.hwPhi
              << " qual: " << std::bitset<4>(egiso.hwQual) << " packed: " << egiso.pack().to_string(16) << std::endl;

  return egobjs.back();
}

void PFTkEGAlgoEmulator::addEgObjsToPF(std::vector<EGObjEmu> &egstas,
                                       std::vector<EGIsoObjEmu> &egobjs,
                                       std::vector<EGIsoEleObjEmu> &egeleobjs,
                                       const std::vector<EmCaloObjEmu> &emcalo,
                                       const std::vector<TkObjEmu> &track,
                                       const int calo_idx,
                                       const unsigned int hwQual,
                                       const pt_t ptCorr,
                                       const int tk_idx,
                                       const float bdtScore,
                                       const std::vector<unsigned int> &components) const {
  int sta_idx = -1;
  if (writeEgSta()) {
    addEGStaToPF(egstas, emcalo[calo_idx], hwQual, ptCorr, components);
    sta_idx = egstas.size() - 1;
  }
  EGIsoObjEmu &egobj = addEGIsoToPF(egobjs, emcalo[calo_idx], hwQual, ptCorr);
  egobj.sta_idx = sta_idx;
  if (tk_idx != -1) {
    EGIsoEleObjEmu &eleobj = addEGIsoEleToPF(egeleobjs, emcalo[calo_idx], track[tk_idx], hwQual, ptCorr, bdtScore);
    eleobj.sta_idx = sta_idx;
  }
}

void PFTkEGAlgoEmulator::runIso(const PFInputRegion &in,
                                const std::vector<l1ct::PVObjEmu> &pvs,
                                OutputRegion &out) const {
  if (cfg.doTkIso) {
    compute_isolation(out.egelectron, in.track, cfg.tkIsoParams_tkEle, pvs[0].hwZ0);
    compute_isolation(out.egphoton, in.track, cfg.tkIsoParams_tkEm, pvs[0].hwZ0);
  }
  if (cfg.doPfIso) {
    compute_isolation(out.egelectron, out.pfcharged, out.pfneutral, cfg.pfIsoParams_tkEle, pvs[0].hwZ0);
    compute_isolation(out.egphoton, out.pfcharged, out.pfneutral, cfg.pfIsoParams_tkEm, pvs[0].hwZ0);
  }

  std::for_each(out.egelectron.begin(), out.egelectron.end(), [&](EGIsoEleObjEmu &obj) {
    obj.hwIso = obj.hwIsoVar(cfg.hwIsoTypeTkEle);
  });
  std::for_each(
      out.egphoton.begin(), out.egphoton.end(), [&](EGIsoObjEmu &obj) { obj.hwIso = obj.hwIsoVar(cfg.hwIsoTypeTkEm); });
}

void PFTkEGAlgoEmulator::compute_isolation(std::vector<EGIsoObjEmu> &egobjs,
                                           const std::vector<TkObjEmu> &objects,
                                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                                           z0_t z0) const {
  for (int ic = 0, nc = egobjs.size(); ic < nc; ++ic) {
    auto &egphoton = egobjs[ic];
    iso_t sumPt = 0.;
    iso_t sumPtPV = 0.;
    compute_sumPt(sumPt, sumPtPV, objects, cfg.nTRACK, egphoton, params, z0);
    egphoton.setHwIso(EGIsoObjEmu::IsoType::TkIso, sumPt);
    egphoton.setHwIso(EGIsoObjEmu::IsoType::TkIsoPV, sumPtPV);
  }
}

void PFTkEGAlgoEmulator::compute_isolation(std::vector<EGIsoEleObjEmu> &egobjs,
                                           const std::vector<TkObjEmu> &objects,
                                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                                           z0_t z0) const {
  for (int ic = 0, nc = egobjs.size(); ic < nc; ++ic) {
    auto &egele = egobjs[ic];
    iso_t sumPt = 0.;
    iso_t sumPtPV = 0.;
    compute_sumPt(sumPt, sumPtPV, objects, cfg.nTRACK, egele, params, z0);
    egele.setHwIso(EGIsoEleObjEmu::IsoType::TkIso, sumPtPV);
  }
}

void PFTkEGAlgoEmulator::compute_isolation(std::vector<EGIsoObjEmu> &egobjs,
                                           const std::vector<PFChargedObjEmu> &charged,
                                           const std::vector<PFNeutralObjEmu> &neutrals,
                                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                                           z0_t z0) const {
  for (int ic = 0, nc = egobjs.size(); ic < nc; ++ic) {
    auto &egphoton = egobjs[ic];
    iso_t sumPt = 0.;
    iso_t sumPtPV = 0.;
    // FIXME: set max # of PF objects for iso
    compute_sumPt(sumPt, sumPtPV, charged, charged.size(), egphoton, params, z0);
    compute_sumPt(sumPt, sumPtPV, neutrals, neutrals.size(), egphoton, params, z0);
    egphoton.setHwIso(EGIsoObjEmu::IsoType::PfIso, sumPt);
    egphoton.setHwIso(EGIsoObjEmu::IsoType::PfIsoPV, sumPtPV);
  }
}

void PFTkEGAlgoEmulator::compute_isolation(std::vector<EGIsoEleObjEmu> &egobjs,
                                           const std::vector<PFChargedObjEmu> &charged,
                                           const std::vector<PFNeutralObjEmu> &neutrals,
                                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                                           z0_t z0) const {
  for (int ic = 0, nc = egobjs.size(); ic < nc; ++ic) {
    auto &egele = egobjs[ic];
    iso_t sumPt = 0.;
    iso_t sumPtPV = 0.;
    compute_sumPt(sumPt, sumPtPV, charged, charged.size(), egele, params, z0);
    compute_sumPt(sumPt, sumPtPV, neutrals, neutrals.size(), egele, params, z0);
    egele.setHwIso(EGIsoEleObjEmu::IsoType::PfIso, sumPtPV);
  }
}
