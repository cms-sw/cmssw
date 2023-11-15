#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/middle_buffer_multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_elements_ref.icc"

#include <iostream>
#include <memory>
#include <stdexcept>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

l1ct::MiddleBufferMultififoRegionizerEmulator::MiddleBufferMultififoRegionizerEmulator(const edm::ParameterSet& iConfig)
    : MiddleBufferMultififoRegionizerEmulator(iConfig.getParameter<uint32_t>("nClocks"),
                                              iConfig.getParameter<uint32_t>("nBuffers"),
                                              iConfig.getParameter<uint32_t>("etaBufferDepth"),
                                              iConfig.getParameter<uint32_t>("nTkLinks"),
                                              iConfig.getParameter<uint32_t>("nHCalLinks"),
                                              iConfig.getParameter<uint32_t>("nECalLinks"),
                                              iConfig.getParameter<uint32_t>("nTrack"),
                                              iConfig.getParameter<uint32_t>("nCalo"),
                                              iConfig.getParameter<uint32_t>("nEmCalo"),
                                              iConfig.getParameter<uint32_t>("nMu"),
                                              /*streaming=*/true,
                                              /*outii=*/2,
                                              /*pauseii=*/1,
                                              iConfig.getParameter<bool>("useAlsoVtxCoords")) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
}

edm::ParameterSetDescription l1ct::MiddleBufferMultififoRegionizerEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<uint32_t>("nClocks", 162);
  description.add<uint32_t>("nBuffers", 27);
  description.add<uint32_t>("etaBufferDepth", 54);
  description.add<uint32_t>("nTkLinks", 1);
  description.add<uint32_t>("nHCalLinks", 1);
  description.add<uint32_t>("nECalLinks", 0);
  description.add<uint32_t>("nTrack", 22);
  description.add<uint32_t>("nCalo", 15);
  description.add<uint32_t>("nEmCalo", 12);
  description.add<uint32_t>("nMu", 2);
  description.add<bool>("useAlsoVtxCoords", true);
  description.addUntracked<bool>("debug", false);
  return description;
}

#endif

l1ct::MiddleBufferMultififoRegionizerEmulator::MiddleBufferMultififoRegionizerEmulator(unsigned int nclocks,
                                                                                       unsigned int nbuffers,
                                                                                       unsigned int etabufferDepth,
                                                                                       unsigned int ntklinks,
                                                                                       unsigned int nHCalLinks,
                                                                                       unsigned int nECalLinks,
                                                                                       unsigned int ntk,
                                                                                       unsigned int ncalo,
                                                                                       unsigned int nem,
                                                                                       unsigned int nmu,
                                                                                       bool streaming,
                                                                                       unsigned int outii,
                                                                                       unsigned int pauseii,
                                                                                       bool useAlsoVtxCoords)
    : RegionizerEmulator(useAlsoVtxCoords),
      NTK_SECTORS(9),
      NCALO_SECTORS(3),
      NTK_LINKS(ntklinks),
      HCAL_LINKS(nHCalLinks),
      ECAL_LINKS(nECalLinks),
      NMU_LINKS(1),
      nclocks_(nclocks),
      nbuffers_(nbuffers),
      etabuffer_depth_(etabufferDepth),
      ntk_(ntk),
      ncalo_(ncalo),
      nem_(nem),
      nmu_(nmu),
      outii_(outii),
      pauseii_(pauseii),
      nregions_pre_(27),
      nregions_post_(54),
      streaming_(streaming),
      init_(false),
      iclock_(0),
      tkRegionizerPre_(ntk, ntk, false, outii, pauseii, useAlsoVtxCoords),
      tkRegionizerPost_(ntk, (ntk + outii - 1) / outii, true, outii, pauseii, useAlsoVtxCoords),
      hadCaloRegionizerPre_(ncalo, ncalo, false, outii, pauseii),
      hadCaloRegionizerPost_(ncalo, (ncalo + outii - 1) / outii, true, outii, pauseii),
      emCaloRegionizerPre_(nem, nem, false, outii, pauseii),
      emCaloRegionizerPost_(nem, (nem + outii - 1) / outii, true, outii, pauseii),
      muRegionizerPre_(nmu, nmu, false, outii, pauseii),
      muRegionizerPost_(nmu, std::max(1u, (nmu + outii - 1) / outii), true, outii, pauseii),
      tkBuffers_(ntk ? nbuffers_ : 0),
      hadCaloBuffers_(ncalo ? nbuffers_ : 0),
      emCaloBuffers_(nem ? nbuffers_ : 0),
      muBuffers_(nmu ? nbuffers_ : 0) {
  assert(nbuffers_ == nregions_post_ || nbuffers_ == nregions_pre_);
  unsigned int phisectors = 9, etaslices = 3;
  for (unsigned int ietaslice = 0; ietaslice < etaslices && ntk > 0; ++ietaslice) {
    for (unsigned int ie = 0; ie < 2; ++ie) {  // 0 = negative, 1 = positive
      unsigned int nTFEtaSlices = ietaslice == 1 ? 2 : 1;
      if ((ietaslice == 0 && ie == 1) || (ietaslice == 2 && ie == 0))
        continue;
      unsigned int ireg0 = phisectors * ietaslice, il0 = 3 * NTK_LINKS * (nTFEtaSlices - 1) * ie;
      for (unsigned int is = 0; is < NTK_SECTORS; ++is) {  // 9 tf sectors
        for (unsigned int il = 0; il < NTK_LINKS; ++il) {  // max tracks per sector per clock
          unsigned int isp = (is + 1) % NTK_SECTORS, ism = (is + NTK_SECTORS - 1) % NTK_SECTORS;
          tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, is + ireg0, il0 + il);
          tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, isp + ireg0, il0 + il + NTK_LINKS);
          tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, ism + ireg0, il0 + il + 2 * NTK_LINKS);
        }
      }
    }
  }
  // calo
  for (unsigned int ie = 0; ie < etaslices; ++ie) {
    for (unsigned int is = 0; is < NCALO_SECTORS; ++is) {  // NCALO_SECTORS sectors
      for (unsigned int j = 0; j < 3; ++j) {               // 3 regions x sector
        for (unsigned int il = 0; il < HCAL_LINKS; ++il) {
          caloRoutes_.emplace_back(is, il, 3 * is + j + phisectors * ie, il);
          if (j) {
            caloRoutes_.emplace_back((is + 1) % 3, il, 3 * is + j + phisectors * ie, il + HCAL_LINKS);
          }
        }
        for (unsigned int il = 0; il < ECAL_LINKS; ++il) {
          emCaloRoutes_.emplace_back(is, il, 3 * is + j + phisectors * ie, il);
          if (j) {
            emCaloRoutes_.emplace_back((is + 1) % 3, il, 3 * is + j + phisectors * ie, il + ECAL_LINKS);
          }
        }
      }
    }
  }
  // mu
  for (unsigned int il = 0; il < NMU_LINKS && nmu > 0; ++il) {
    for (unsigned int j = 0; j < nregions_pre_; ++j) {
      muRoutes_.emplace_back(0, il, j, il);
    }
  }
}

l1ct::MiddleBufferMultififoRegionizerEmulator::~MiddleBufferMultififoRegionizerEmulator() {}

void l1ct::MiddleBufferMultififoRegionizerEmulator::initSectorsAndRegions(const RegionizerDecodedInputs& in,
                                                                          const std::vector<PFInputRegion>& out) {
  assert(!init_);
  init_ = true;
  assert(out.size() == nregions_post_);

  std::vector<PFInputRegion> mergedRegions;
  unsigned int neta = 3, nphi = 9;
  mergedRegions.reserve(nregions_pre_);
  mergedRegions_.reserve(nregions_pre_);
  outputRegions_.reserve(nregions_post_);
  for (unsigned int ieta = 0; ieta < neta; ++ieta) {
    for (unsigned int iphi = 0; iphi < nphi; ++iphi) {
      const PFRegionEmu& reg0 = out[(2 * ieta + 0) * nphi + iphi].region;
      const PFRegionEmu& reg1 = out[(2 * ieta + 1) * nphi + iphi].region;
      assert(reg0.hwPhiCenter == reg1.hwPhiCenter);
      mergedRegions.emplace_back(reg0.floatEtaMin(),
                                 reg1.floatEtaMax(),
                                 reg0.floatPhiCenter(),
                                 reg0.floatPhiHalfWidth() * 2,
                                 reg0.floatEtaExtra(),
                                 reg0.floatPhiExtra());
      mergedRegions_.push_back(mergedRegions.back().region);
      outputRegions_.push_back(reg0);
      outputRegions_.push_back(reg1);
      if (debug_) {
        dbgCout() << "Created region with etaCenter " << mergedRegions.back().region.hwEtaCenter.to_int()
                  << ", halfWidth " << mergedRegions.back().region.hwEtaHalfWidth.to_int() << "\n";
      }
      if (nbuffers_ == nregions_post_) {
        for (int i = 0; i < 2; ++i) {
          unsigned int iout = (2 * ieta + i) * nphi + iphi;
          const l1ct::PFRegionEmu& from = mergedRegions.back().region;
          const l1ct::PFRegionEmu& to = out[iout].region;
          l1ct::glbeta_t etaMin = to.hwEtaCenter - to.hwEtaHalfWidth - to.hwEtaExtra - from.hwEtaCenter;
          l1ct::glbeta_t etaMax = to.hwEtaCenter + to.hwEtaHalfWidth + to.hwEtaExtra - from.hwEtaCenter;
          l1ct::glbeta_t etaShift = from.hwEtaCenter - to.hwEtaCenter;
          l1ct::glbphi_t phiMin = -to.hwPhiHalfWidth - to.hwPhiExtra;
          l1ct::glbphi_t phiMax = +to.hwPhiHalfWidth + to.hwPhiExtra;
          l1ct::glbphi_t phiShift = 0;
          if (ntk_ > 0)
            tkBuffers_[iout] = l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::TkObjEmu>(
                etabuffer_depth_, etaMin, etaMax, etaShift, phiMin, phiMax, phiShift);
          if (ncalo_ > 0)
            hadCaloBuffers_[iout] = l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::HadCaloObjEmu>(
                etabuffer_depth_, etaMin, etaMax, etaShift, phiMin, phiMax, phiShift);
          if (nem_ > 0)
            emCaloBuffers_[iout] = l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::EmCaloObjEmu>(
                etabuffer_depth_, etaMin, etaMax, etaShift, phiMin, phiMax, phiShift);
          if (nmu_ > 0)
            muBuffers_[iout] = l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::MuObjEmu>(
                etabuffer_depth_, etaMin, etaMax, etaShift, phiMin, phiMax, phiShift);
        }
      } else if (nbuffers_ == nregions_pre_) {
        unsigned int iout = ieta * nphi + iphi;
        if (ntk_ > 0)
          tkBuffers_[iout] = l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::TkObjEmu>(etabuffer_depth_);
        if (ncalo_ > 0)
          hadCaloBuffers_[iout] = l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::HadCaloObjEmu>(etabuffer_depth_);
        if (nem_ > 0)
          emCaloBuffers_[iout] = l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::EmCaloObjEmu>(etabuffer_depth_);
        if (nmu_ > 0)
          muBuffers_[iout] = l1ct::multififo_regionizer::EtaPhiBuffer<l1ct::MuObjEmu>(etabuffer_depth_);
      }
    }
  }
  if (ntk_) {
    assert(in.track.size() == 2 * NTK_SECTORS);
    tkRegionizerPre_.initSectors(in.track);
    tkRegionizerPre_.initRegions(mergedRegions);
    tkRegionizerPre_.initRouting(tkRoutes_);
    tkRegionizerPost_.initRegions(out);
  }
  if (ncalo_) {
    assert(in.hadcalo.size() == NCALO_SECTORS);
    hadCaloRegionizerPre_.initSectors(in.hadcalo);
    hadCaloRegionizerPre_.initRegions(mergedRegions);
    hadCaloRegionizerPre_.initRouting(caloRoutes_);
    hadCaloRegionizerPost_.initRegions(out);
  }
  if (nem_) {
    assert(in.emcalo.size() == NCALO_SECTORS);
    emCaloRegionizerPre_.initSectors(in.emcalo);
    emCaloRegionizerPre_.initRegions(mergedRegions);
    if (ECAL_LINKS)
      emCaloRegionizerPre_.initRouting(emCaloRoutes_);
    emCaloRegionizerPost_.initRegions(out);
  }
  if (nmu_) {
    muRegionizerPre_.initSectors(in.muon);
    muRegionizerPre_.initRegions(mergedRegions);
    muRegionizerPre_.initRouting(muRoutes_);
    muRegionizerPost_.initRegions(out);
  }
}

bool l1ct::MiddleBufferMultififoRegionizerEmulator::step(bool newEvent,
                                                         const std::vector<l1ct::TkObjEmu>& links_tk,
                                                         const std::vector<l1ct::HadCaloObjEmu>& links_hadCalo,
                                                         const std::vector<l1ct::EmCaloObjEmu>& links_emCalo,
                                                         const std::vector<l1ct::MuObjEmu>& links_mu,
                                                         std::vector<l1ct::TkObjEmu>& out_tk,
                                                         std::vector<l1ct::HadCaloObjEmu>& out_hadCalo,
                                                         std::vector<l1ct::EmCaloObjEmu>& out_emCalo,
                                                         std::vector<l1ct::MuObjEmu>& out_mu,
                                                         bool /*unused*/) {
  iclock_ = (newEvent ? 0 : iclock_ + 1);
  bool newRead = iclock_ == 2 * etabuffer_depth_;

  std::vector<l1ct::TkObjEmu> pre_out_tk;
  std::vector<l1ct::HadCaloObjEmu> pre_out_hadCalo;
  std::vector<l1ct::EmCaloObjEmu> pre_out_emCalo;
  std::vector<l1ct::MuObjEmu> pre_out_mu;
  bool ret = false;
  if (ntk_)
    ret = tkRegionizerPre_.step(newEvent, links_tk, pre_out_tk, false);
  if (nmu_)
    ret = muRegionizerPre_.step(newEvent, links_mu, pre_out_mu, false);
  if (ncalo_)
    ret = hadCaloRegionizerPre_.step(newEvent, links_hadCalo, pre_out_hadCalo, false);
  if (nem_) {
    if (ECAL_LINKS) {
      ret = emCaloRegionizerPre_.step(newEvent, links_emCalo, pre_out_emCalo, false);
    } else if (ncalo_) {
      pre_out_emCalo.resize(pre_out_hadCalo.size());
      for (unsigned int i = 0, n = pre_out_hadCalo.size(); i < n; ++i) {
        decode(pre_out_hadCalo[i], pre_out_emCalo[i]);
      }
    }
  }

  // in the no-streaming case, we just output the pre-regionizer
  if (!streaming_) {
    out_tk.swap(pre_out_tk);
    out_mu.swap(pre_out_mu);
    out_hadCalo.swap(pre_out_hadCalo);
    out_emCalo.swap(pre_out_emCalo);
    return ret;
  }

  // otherwise, we push into the eta buffers
  if (newEvent) {
    for (auto& b : tkBuffers_)
      b.writeNewEvent();
    for (auto& b : hadCaloBuffers_)
      b.writeNewEvent();
    for (auto& b : emCaloBuffers_)
      b.writeNewEvent();
    for (auto& b : muBuffers_)
      b.writeNewEvent();
  }
  unsigned int neta = 3, nphi = 9;
  for (unsigned int ieta = 0; ieta < neta; ++ieta) {
    for (unsigned int iphi = 0; iphi < nphi; ++iphi) {
      unsigned int iin = ieta * nphi + iphi;
      for (int i = 0, n = nbuffers_ == nregions_pre_ ? 1 : 2; i < n; ++i) {
        unsigned int iout = (n * ieta + i) * nphi + iphi;
        if (ntk_)
          tkBuffers_[iout].maybe_push(pre_out_tk[iin]);
        if (ncalo_)
          hadCaloBuffers_[iout].maybe_push(pre_out_hadCalo[iin]);
        if (nem_)
          emCaloBuffers_[iout].maybe_push(pre_out_emCalo[iin]);
        if (nmu_)
          muBuffers_[iout].maybe_push(pre_out_mu[iin]);
      }
    }
  }

  // and we read from eta buffers into muxes
  if (newRead) {
    for (auto& b : tkBuffers_)
      b.readNewEvent();
    for (auto& b : hadCaloBuffers_)
      b.readNewEvent();
    for (auto& b : emCaloBuffers_)
      b.readNewEvent();
    for (auto& b : muBuffers_)
      b.readNewEvent();
  }
  std::vector<l1ct::TkObjEmu> bufferOut_tk(ntk_ ? nregions_post_ : 0);
  std::vector<l1ct::HadCaloObjEmu> bufferOut_hadCalo(ncalo_ ? nregions_post_ : 0);
  std::vector<l1ct::EmCaloObjEmu> bufferOut_emCalo(nem_ ? nregions_post_ : 0);
  std::vector<l1ct::MuObjEmu> bufferOut_mu(nmu_ ? nregions_post_ : 0);
  if (nbuffers_ == nregions_post_) {  // just copy directly
    for (unsigned int i = 0; i < nregions_post_; ++i) {
      if (ntk_)
        bufferOut_tk[i] = tkBuffers_[i].pop();
      if (ncalo_)
        bufferOut_hadCalo[i] = hadCaloBuffers_[i].pop();
      if (nem_)
        bufferOut_emCalo[i] = emCaloBuffers_[i].pop();
      if (nmu_)
        bufferOut_mu[i] = muBuffers_[i].pop();
    }
  } else if (nbuffers_ == nregions_pre_) {  // propagate and copy
    unsigned int neta = 3, nphi = 9;
    for (unsigned int ieta = 0; ieta < neta; ++ieta) {
      for (unsigned int iphi = 0; iphi < nphi; ++iphi) {
        unsigned int iin = ieta * nphi + iphi;
        const l1ct::PFRegionEmu& from = mergedRegions_[iin];
        l1ct::TkObjEmu tk = ntk_ ? tkBuffers_[iin].pop() : l1ct::TkObjEmu();
        l1ct::HadCaloObjEmu calo = ncalo_ ? hadCaloBuffers_[iin].pop() : l1ct::HadCaloObjEmu();
        l1ct::EmCaloObjEmu em = nem_ ? emCaloBuffers_[iin].pop() : l1ct::EmCaloObjEmu();
        l1ct::MuObjEmu mu = nmu_ ? muBuffers_[iin].pop() : l1ct::MuObjEmu();
        for (int i = 0; i < 2; ++i) {
          const l1ct::PFRegionEmu& to = outputRegions_[2 * iin + i];
          unsigned int iout = (2 * ieta + i) * nphi + iphi;
          l1ct::glbeta_t etaMin = to.hwEtaCenter - to.hwEtaHalfWidth - to.hwEtaExtra - from.hwEtaCenter;
          l1ct::glbeta_t etaMax = to.hwEtaCenter + to.hwEtaHalfWidth + to.hwEtaExtra - from.hwEtaCenter;
          l1ct::glbeta_t etaShift = from.hwEtaCenter - to.hwEtaCenter;
          l1ct::glbphi_t phiMin = -to.hwPhiHalfWidth - to.hwPhiExtra;
          l1ct::glbphi_t phiMax = +to.hwPhiHalfWidth + to.hwPhiExtra;
          if (tk.hwPt > 0 && l1ct::multififo_regionizer::local_eta_phi_window(tk, etaMin, etaMax, phiMin, phiMax)) {
            bufferOut_tk[iout] = tk;
            bufferOut_tk[iout].hwEta += etaShift;
          }
          if (calo.hwPt > 0 && l1ct::multififo_regionizer::local_eta_phi_window(calo, etaMin, etaMax, phiMin, phiMax)) {
            bufferOut_hadCalo[iout] = calo;
            bufferOut_hadCalo[iout].hwEta += etaShift;
          }
          if (em.hwPt > 0 && l1ct::multififo_regionizer::local_eta_phi_window(em, etaMin, etaMax, phiMin, phiMax)) {
            bufferOut_emCalo[iout] = em;
            bufferOut_emCalo[iout].hwEta += etaShift;
          }
          if (mu.hwPt > 0 && l1ct::multififo_regionizer::local_eta_phi_window(mu, etaMin, etaMax, phiMin, phiMax)) {
            bufferOut_mu[iout] = mu;
            bufferOut_mu[iout].hwEta += etaShift;
          }
        }
      }
    }
  }
  if (ntk_)
    tkRegionizerPost_.muxonly_step(newEvent, /*flush=*/true, bufferOut_tk, out_tk);
  if (ncalo_)
    hadCaloRegionizerPost_.muxonly_step(newEvent, /*flush=*/true, bufferOut_hadCalo, out_hadCalo);
  if (nem_)
    emCaloRegionizerPost_.muxonly_step(newEvent, /*flush=*/true, bufferOut_emCalo, out_emCalo);
  if (nmu_)
    muRegionizerPost_.muxonly_step(newEvent, /*flush=*/true, bufferOut_mu, out_mu);

  return newRead;
}

void l1ct::MiddleBufferMultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                              const l1ct::RegionizerDecodedInputs& in,
                                                              std::vector<l1ct::TkObjEmu>& links,
                                                              std::vector<bool>& valid) {
  if (ntk_ == 0)
    return;
  assert(NTK_LINKS == 1);
  links.resize(NTK_SECTORS * NTK_LINKS * 2);
  valid.resize(links.size());
  // emulate reduced rate from 96b tracks on 64b links
  unsigned int itkclock = 2 * (iclock / 3) + (iclock % 3) - 1;  // will underflow for iclock == 0 but it doesn't matter
  for (unsigned int is = 0, idx = 0; is < 2 * NTK_SECTORS; ++is, ++idx) {  // tf sectors
    const l1ct::DetectorSector<l1ct::TkObjEmu>& sec = in.track[is];
    unsigned int ntracks = sec.size();
    unsigned int nw64 = (ntracks * 3 + 1) / 2;
    if (iclock % 3 == 0) {
      links[idx].clear();
      valid[idx] = (iclock == 0) || (iclock < nw64);
    } else if (itkclock < ntracks && itkclock < nclocks_ - 1) {
      links[idx] = sec[itkclock];
      valid[idx] = true;
    } else {
      links[idx].clear();
      valid[idx] = false;
    }
  }
}

template <typename T>
void l1ct::MiddleBufferMultififoRegionizerEmulator::fillCaloLinks_(unsigned int iclock,
                                                                   const std::vector<DetectorSector<T>>& in,
                                                                   std::vector<T>& links,
                                                                   std::vector<bool>& valid) {
  unsigned int NLINKS = (typeid(T) == typeid(l1ct::HadCaloObjEmu) ? HCAL_LINKS : ECAL_LINKS);
  links.resize(NCALO_SECTORS * NLINKS);
  valid.resize(links.size());
  for (unsigned int is = 0, idx = 0; is < NCALO_SECTORS; ++is) {
    for (unsigned int il = 0; il < NLINKS; ++il, ++idx) {
      unsigned int ioffs = iclock * NLINKS + il;
      if (ioffs < in[is].size() && iclock < nclocks_ - 1) {
        links[idx] = in[is][ioffs];
        valid[idx] = true;
      } else {
        links[idx].clear();
        valid[idx] = false;
      }
    }
  }
}
void l1ct::MiddleBufferMultififoRegionizerEmulator::fillSharedCaloLinks(
    unsigned int iclock,
    const std::vector<l1ct::DetectorSector<l1ct::EmCaloObjEmu>>& em_in,
    const std::vector<l1ct::DetectorSector<l1ct::HadCaloObjEmu>>& had_in,
    std::vector<l1ct::HadCaloObjEmu>& links,
    std::vector<bool>& valid) {
  assert(ECAL_LINKS == 0 && HCAL_LINKS == 1 && ncalo_ != 0 && nem_ != 0);
  links.resize(NCALO_SECTORS);
  valid.resize(links.size());
  // for the moment we assume the first 54 clocks are for EM, the rest for HAD
  const unsigned int NCLK_EM = 54;
  for (unsigned int is = 0; is < NCALO_SECTORS; ++is) {
    links[is].clear();
    if (iclock < NCLK_EM) {
      valid[is] = true;
      if (iclock < em_in[is].size()) {
        encode(em_in[is][iclock], links[is]);
      }
    } else {
      if (iclock - NCLK_EM < had_in[is].size()) {
        encode(had_in[is][iclock - NCLK_EM], links[is]);
        valid[is] = true;
      } else {
        valid[is] = false;
      }
    }
  }  // sectors
}

void l1ct::MiddleBufferMultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                              const l1ct::RegionizerDecodedInputs& in,
                                                              std::vector<l1ct::HadCaloObjEmu>& links,
                                                              std::vector<bool>& valid) {
  if (ncalo_ == 0)
    return;
  if (nem_ != 0 && ECAL_LINKS == 0 && HCAL_LINKS == 1)
    fillSharedCaloLinks(iclock, in.emcalo, in.hadcalo, links, valid);
  else
    fillCaloLinks_(iclock, in.hadcalo, links, valid);
}

void l1ct::MiddleBufferMultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                              const l1ct::RegionizerDecodedInputs& in,
                                                              std::vector<l1ct::EmCaloObjEmu>& links,
                                                              std::vector<bool>& valid) {
  if (nem_ == 0)
    return;
  fillCaloLinks_(iclock, in.emcalo, links, valid);
}

void l1ct::MiddleBufferMultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                              const l1ct::RegionizerDecodedInputs& in,
                                                              std::vector<l1ct::MuObjEmu>& links,
                                                              std::vector<bool>& valid) {
  if (nmu_ == 0)
    return;
  assert(NMU_LINKS == 1);
  links.resize(NMU_LINKS);
  valid.resize(links.size());
  if (iclock < in.muon.size() && iclock < nclocks_ - 1) {
    links[0] = in.muon[iclock];
    valid[0] = true;
  } else {
    links[0].clear();
    valid[0] = false;
  }
}

void l1ct::MiddleBufferMultififoRegionizerEmulator::destream(int iclock,
                                                             const std::vector<l1ct::TkObjEmu>& tk_out,
                                                             const std::vector<l1ct::EmCaloObjEmu>& em_out,
                                                             const std::vector<l1ct::HadCaloObjEmu>& calo_out,
                                                             const std::vector<l1ct::MuObjEmu>& mu_out,
                                                             PFInputRegion& out) {
  if (ntk_)
    tkRegionizerPost_.destream(iclock, tk_out, out.track);
  if (ncalo_)
    hadCaloRegionizerPost_.destream(iclock, calo_out, out.hadcalo);
  if (nem_)
    emCaloRegionizerPost_.destream(iclock, em_out, out.emcalo);
  if (nmu_)
    muRegionizerPost_.destream(iclock, mu_out, out.muon);
}

void l1ct::MiddleBufferMultififoRegionizerEmulator::reset() {
  tkRegionizerPre_.reset();
  emCaloRegionizerPre_.reset();
  hadCaloRegionizerPre_.reset();
  muRegionizerPre_.reset();
  tkRegionizerPost_.reset();
  emCaloRegionizerPost_.reset();
  hadCaloRegionizerPost_.reset();
  muRegionizerPost_.reset();
  for (auto& b : tkBuffers_)
    b.reset();
  for (auto& b : hadCaloBuffers_)
    b.reset();
  for (auto& b : emCaloBuffers_)
    b.reset();
  for (auto& b : muBuffers_)
    b.reset();
}

void l1ct::MiddleBufferMultififoRegionizerEmulator::run(const RegionizerDecodedInputs& in,
                                                        std::vector<PFInputRegion>& out) {
  assert(streaming_);  // doesn't make sense otherwise
  if (!init_)
    initSectorsAndRegions(in, out);
  reset();
  std::vector<l1ct::TkObjEmu> tk_links_in, tk_out;
  std::vector<l1ct::EmCaloObjEmu> em_links_in, em_out;
  std::vector<l1ct::HadCaloObjEmu> calo_links_in, calo_out;
  std::vector<l1ct::MuObjEmu> mu_links_in, mu_out;

  // read and sort the inputs
  for (unsigned int iclock = 0; iclock < nclocks_; ++iclock) {
    fillLinks(iclock, in, tk_links_in);
    fillLinks(iclock, in, em_links_in);
    fillLinks(iclock, in, calo_links_in);
    fillLinks(iclock, in, mu_links_in);

    bool newevt = (iclock == 0);
    step(newevt, tk_links_in, calo_links_in, em_links_in, mu_links_in, tk_out, calo_out, em_out, mu_out, true);
  }

  // set up an empty event
  for (auto& l : tk_links_in)
    l.clear();
  for (auto& l : em_links_in)
    l.clear();
  for (auto& l : calo_links_in)
    l.clear();
  for (auto& l : mu_links_in)
    l.clear();

  // read and put the inputs in the regions
  assert(out.size() == nregions_post_);
  for (unsigned int iclock = 0; iclock < nclocks_; ++iclock) {
    bool newevt = (iclock == 0);
    step(newevt, tk_links_in, calo_links_in, em_links_in, mu_links_in, tk_out, calo_out, em_out, mu_out, true);

    unsigned int ireg = (iclock / (outii_ + pauseii_));
    if ((iclock % (outii_ + pauseii_)) >= outii_)
      continue;
    if (ireg >= nregions_post_)
      break;

    if (streaming_) {
      destream(iclock, tk_out, em_out, calo_out, mu_out, out[ireg]);
    } else {
      if (iclock % outii_ == 0) {
        out[ireg].track = tk_out;
        out[ireg].emcalo = em_out;
        out[ireg].hadcalo = calo_out;
        out[ireg].muon = mu_out;
      }
    }
  }

  reset();
}

void l1ct::MiddleBufferMultififoRegionizerEmulator::encode(const l1ct::EmCaloObjEmu& from, l1ct::HadCaloObjEmu& to) {
  assert(!from.hwEmID[5]);
  to.hwPt = from.hwPt;
  to.hwEmPt = from.hwPtErr;
  to.hwEta = from.hwEta;
  to.hwPhi = from.hwPhi;
  to.hwEmID[5] = true;
  to.hwEmID(4, 0) = from.hwEmID(4, 0);
  to.src = from.src;
}
void l1ct::MiddleBufferMultififoRegionizerEmulator::encode(const l1ct::HadCaloObjEmu& from, l1ct::HadCaloObjEmu& to) {
  assert(!from.hwEmID[5]);
  to = from;
}
void l1ct::MiddleBufferMultififoRegionizerEmulator::decode(l1ct::HadCaloObjEmu& had, l1ct::EmCaloObjEmu& em) {
  if (had.hwPt && had.hwEmID[5]) {
    em.hwPt = had.hwPt;
    em.hwPtErr = had.hwEmPt;
    em.hwEta = had.hwEta;
    em.hwPhi = had.hwPhi;
    em.hwEmID[5] = 0;
    em.hwEmID(4, 0) = had.hwEmID(4, 0);
    em.hwSrrTot = 0;
    em.hwMeanZ = 0;
    em.hwHoe = 0;
    em.src = had.src;
    had.clear();
  } else {
    em.clear();
  }
}