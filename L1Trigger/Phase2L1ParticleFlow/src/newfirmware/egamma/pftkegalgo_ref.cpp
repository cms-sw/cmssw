#include "pftkegalgo_ref.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <iostream>

using namespace l1ct;


#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::pftkegalgo_config::pftkegalgo_config(const edm::ParameterSet & pset) :
  nTRACK(pset.getParameter<uint32_t>("nTRACK")), 
  nEMCALO(pset.getParameter<uint32_t>("nEMCALO")), 
  nEMCALOSEL_EGIN(pset.getParameter<uint32_t>("nEMCALOSEL_EGIN")), 
  nEM_EGOUT(pset.getParameter<uint32_t>("nEM_EGOUT")),
  filterHwQuality(pset.getParameter<bool>("filterHwQuality")),
  doBremRecovery(pset.getParameter<bool>("doBremRecovery")),
  caloHwQual(pset.getParameter<int>("caloHwQual")),
  emClusterPtMin(pset.getParameter<double>("caloEtMin")),
  dEtaMaxBrem(pset.getParameter<double>("dEtaMaxBrem")),
  dPhiMaxBrem(pset.getParameter<double>("dPhiMaxBrem")),
  absEtaBoundaries(pset.getParameter<std::vector<double>>("absEtaBoundaries")),
  dEtaValues(pset.getParameter<std::vector<double>>("dEtaValues")),
  dPhiValues(pset.getParameter<std::vector<double>>("dPhiValues")),
  trkQualityPtMin(pset.getParameter<double>("trkQualityPtMin")) {
}
    
#endif



void PFTkEGAlgoEmulator::toFirmware(const PFInputRegion & in, PFRegion & region, EmCaloObj emcalo[/*nCALO*/], TkObj track[/*nTRACK*/]) const {
    region = in.region;
    l1ct::toFirmware(in.track, cfg.nTRACK, track);
    l1ct::toFirmware(in.emcalo, cfg.nEMCALO, emcalo);
}

void PFTkEGAlgoEmulator::toFirmware(const OutputRegion & out, EGIsoObj out_egphs[], EGIsoEleObj out_egeles[]) const {
    l1ct::toFirmware(out.egphoton, cfg.nEM_EGOUT, out_egphs);
    l1ct::toFirmware(out.egelectron, cfg.nEM_EGOUT, out_egeles);
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

void PFTkEGAlgoEmulator::link_emCalo2emCalo(const std::vector<EmCaloObjEmu>& emcalo,
                                            std::vector<int> &emCalo2emCalo) const {
  // NOTE: we assume the input to be sorted!!!
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];
    if (cfg.filterHwQuality && calo.hwFlags != cfg.caloHwQual)
      continue;

    if (emCalo2emCalo[ic] != -1)
      continue;

    for (int jc = ic + 1; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] != -1)
        continue;

      auto &otherCalo = emcalo[jc];
      if (cfg.filterHwQuality && otherCalo.hwFlags != cfg.caloHwQual)
        continue;

      if (fabs(otherCalo.floatEta() - calo.floatEta()) < cfg.dEtaMaxBrem &&
          fabs(deltaPhi(otherCalo.floatPhi(), calo.floatPhi())) < cfg.dPhiMaxBrem) {
        emCalo2emCalo[jc] = ic;
      }
    }
  }
}



void PFTkEGAlgoEmulator::link_emCalo2tk(const PFRegionEmu &r,
                                        const std::vector<EmCaloObjEmu> &emcalo,
                                        const std::vector<TkObjEmu> &track,
                                        std::vector<int> &emCalo2tk) const {
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];

    if (cfg.filterHwQuality && calo.hwFlags != cfg.caloHwQual)
      continue;

    float dPtMin = 999;
    for (int itk = 0, ntk = track.size(); itk < ntk; ++itk) {
      const auto &tk = track[itk];
      if (tk.floatPt() < cfg.trkQualityPtMin)
        continue;

      float d_phi = deltaPhi(tk.floatPhi(), calo.floatPhi());
      float d_eta = tk.floatEta() - calo.floatEta();  // We only use it squared

      auto eta_index =
          std::distance(
              cfg.absEtaBoundaries.begin(),
              std::lower_bound(cfg.absEtaBoundaries.begin(), cfg.absEtaBoundaries.end(), abs(r.floatGlbEta(calo.hwEta)))) -
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

void PFTkEGAlgoEmulator::sel_emCalo(unsigned int nmax_sel,
                                    const std::vector<EmCaloObjEmu> &emcalo, 
                                    std::vector<EmCaloObjEmu> &emcalo_sel) const {
    std::vector<EmCaloObjEmu> emcalo_tmp;

    for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
      const auto &calo = emcalo[ic];
      if(debug_ > 5)
        std::cout << "[REF] IN calo[" << ic << "] pt: " << calo.hwPt << " eta: " << calo.hwEta << " phi: " << calo.hwPhi << std::endl;
      if ((cfg.filterHwQuality && calo.hwFlags != cfg.caloHwQual) || (calo.floatPt() < cfg.emClusterPtMin)) continue;
      emcalo_tmp.push_back(calo);
    }
    unsigned int nout   = std::min<unsigned>(nmax_sel, emcalo_tmp.size());
    ptsort_ref(emcalo_tmp.size(), nout, emcalo_tmp, emcalo_sel);
}
                                  
void PFTkEGAlgoEmulator::run(const PFInputRegion & in, OutputRegion & out) const {
  
  // FIXME: this is not striclty speaking necessary but we have to avoid sorting differences
  // in the future we will do all the filtering upstream for the endcap
  std::vector<EmCaloObjEmu> emcalo_sel;
  sel_emCalo(cfg.nEMCALOSEL_EGIN, in.emcalo, emcalo_sel);

  std::vector<int> emCalo2emCalo(emcalo_sel.size(), -1);
  if (cfg.doBremRecovery)
    link_emCalo2emCalo(emcalo_sel, emCalo2emCalo);

  std::vector<int> emCalo2tk(emcalo_sel.size(), -1);
  link_emCalo2tk(in.region, emcalo_sel, in.track, emCalo2tk);


  std::vector<EGIsoObjEmu> egobjs;
  std::vector<EGIsoEleObjEmu> egeleobjs;
  eg_algo(emcalo_sel, in.track, emCalo2emCalo, emCalo2tk, egobjs, egeleobjs);

  
  unsigned int nEGOut   = std::min<unsigned>(cfg.nEM_EGOUT, egobjs.size());
  unsigned int nEGEleOut   = std::min<unsigned>(cfg.nEM_EGOUT, egeleobjs.size());

  // init output containers
  out.egphoton.clear();
  out.egelectron.clear();  
  ptsort_ref(egobjs.size(), nEGOut, egobjs, out.egphoton);
  ptsort_ref(egeleobjs.size(), nEGEleOut, egeleobjs, out.egelectron);
  
}


void PFTkEGAlgoEmulator::eg_algo(const std::vector<EmCaloObjEmu> &emcalo,
                                 const std::vector<TkObjEmu> &track,
                                 const std::vector<int> &emCalo2emCalo,
                                 const std::vector<int> &emCalo2tk,
                                 std::vector<EGIsoObjEmu> &egobjs,
                                 std::vector<EGIsoEleObjEmu> &egeleobjs) const {
               
  for (int ic = 0, nc = emcalo.size(); ic < nc; ++ic) {
    auto &calo = emcalo[ic];

    if (cfg.filterHwQuality && calo.hwFlags != cfg.caloHwQual)
      continue;

    if(debug_>1)
      std::cout << "[REF] SEL emcalo with pt: " << calo.hwPt << " qual: " << calo.hwFlags << " eta: " << calo.hwEta << " phi " << calo.hwPhi << std::endl;

    int itk = emCalo2tk[ic];


    // check if brem recovery is on
    if (!cfg.doBremRecovery) {
      // 1. create EG objects before brem recovery
      addEgObjsToPF(emcalo, track, ic, calo.hwFlags, calo.hwPt, itk, egobjs, egeleobjs);
      continue;
    }

    // check if the cluster has already been used in a brem reclustering
    if (emCalo2emCalo[ic] != -1)
      continue;

    pt_t ptBremReco = calo.hwPt;

    for (int jc = ic; jc < nc; ++jc) {
      if (emCalo2emCalo[jc] == ic) {
        auto &otherCalo = emcalo[jc];
        ptBremReco += otherCalo.hwPt;
      }
    }

    // 2. create EG objects with brem recovery
    // FIXME: duplicating the object is suboptimal but this is done for keeping things as in TDR code...
    addEgObjsToPF(emcalo, track, ic, calo.hwFlags+1, ptBremReco, itk, egobjs, egeleobjs);
  }
}

EGIsoObjEmu &PFTkEGAlgoEmulator::addEGIsoToPF(std::vector<EGIsoObjEmu> &egobjs,
                                        const EmCaloObjEmu &calo,
                                        const int hwQual,
                                        const pt_t ptCorr) const {
  if(debug_>1)
    std::cout << "[REF] Add EGIsoObjEmu with pt: " << ptCorr << " qual: " << hwQual << " eta: " << calo.hwEta << " phi " << calo.hwPhi << std::endl;

  // FIXME: content
  EGIsoObjEmu egiso;
  egiso.hwPt = ptCorr;
  egiso.hwEta = calo.hwEta;
  egiso.hwPhi = calo.hwPhi;
  // egiso.hwQual = hwQual;
  // egiso.hwIso = 0;
  egobjs.push_back(egiso);
  return egobjs.back();
}


EGIsoEleObjEmu &PFTkEGAlgoEmulator::addEGIsoEleToPF(std::vector<EGIsoEleObjEmu> &egobjs,
                                                    const EmCaloObjEmu &calo,
                                                    const TkObjEmu &track,
                                                    const int hwQual,
                                                    const pt_t ptCorr) const {
  if(debug_>1)
    std::cout << "[REF] Add EGIsoEleObjEmu with pt: " << ptCorr << " qual: " << hwQual << " eta: " << calo.hwEta << " phi " << calo.hwPhi << std::endl;

  // FIXME: event content
  EGIsoEleObjEmu egiso;
  egiso.hwPt = ptCorr;
  egiso.hwEta = calo.hwEta;
  egiso.hwPhi = calo.hwPhi;

  // egiso.hwVtxEta = track.hwVtxEta;
  // egiso.hwVtxPhi = track.hwVtxPhi;
  egiso.hwZ0 = track.hwZ0;
  // egiso.hwCharge = track.hwCharge;

  // egiso.hwQual = hwQual;
  // egiso.hwIso = 0;
  egobjs.push_back(egiso);
  return egobjs.back();
}


void PFTkEGAlgoEmulator::addEgObjsToPF(const std::vector<EmCaloObjEmu> &emcalo,
                                       const std::vector<TkObjEmu> &track,
                                       const int calo_idx, 
                                       const int hwQual, 
                                       const pt_t ptCorr, 
                                       const int tk_idx,
                                       std::vector<EGIsoObjEmu> &egobjs,
                                       std::vector<EGIsoEleObjEmu> &egeleobjs) const {
  EGIsoObjEmu &egobj = addEGIsoToPF(egobjs, emcalo[calo_idx], hwQual, ptCorr);
  if (tk_idx != -1) {
    addEGIsoEleToPF(egeleobjs, emcalo[calo_idx], track[tk_idx], hwQual, ptCorr);
  }
}
