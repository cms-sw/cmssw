#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/egamma/pfeginput_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/multififo_regionizer_elements_ref.icc"

#include <iostream>
#include <memory>
#include <stdexcept>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::MultififoRegionizerEmulator::MultififoRegionizerEmulator(const edm::ParameterSet& iConfig)
    : MultififoRegionizerEmulator(iConfig.getParameter<uint32_t>("nEndcaps"),
                                  iConfig.getParameter<uint32_t>("nClocks"),
                                  iConfig.getParameter<uint32_t>("nTkLinks"),
                                  iConfig.getParameter<uint32_t>("nCaloLinks"),
                                  iConfig.getParameter<uint32_t>("nTrack"),
                                  iConfig.getParameter<uint32_t>("nCalo"),
                                  iConfig.getParameter<uint32_t>("nEmCalo"),
                                  iConfig.getParameter<uint32_t>("nMu"),
                                  /*streaming=*/false,
                                  /*outii=*/1,
                                  /*pauseii=*/0,
                                  iConfig.getParameter<bool>("useAlsoVtxCoords")) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  if (iConfig.existsAs<edm::ParameterSet>("egInterceptMode")) {
    const auto& emSelCfg = iConfig.getParameter<edm::ParameterSet>("egInterceptMode");
    setEgInterceptMode(emSelCfg.getParameter<bool>("afterFifo"), emSelCfg);
  }
}

l1ct::MultififoRegionizerEmulator::MultififoRegionizerEmulator(const std::string& barrelSetup,
                                                               const edm::ParameterSet& iConfig)
    : MultififoRegionizerEmulator(parseBarrelSetup(barrelSetup),
                                  iConfig.getParameter<uint32_t>("nHCalLinks"),
                                  iConfig.getParameter<uint32_t>("nECalLinks"),
                                  iConfig.getParameter<uint32_t>("nClocks"),
                                  iConfig.getParameter<uint32_t>("nTrack"),
                                  iConfig.getParameter<uint32_t>("nCalo"),
                                  iConfig.getParameter<uint32_t>("nEmCalo"),
                                  iConfig.getParameter<uint32_t>("nMu"),
                                  /*streaming=*/false,
                                  /*outii=*/1,
                                  /*pauseii=*/0,
                                  iConfig.getParameter<bool>("useAlsoVtxCoords")) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
}
#endif

l1ct::MultififoRegionizerEmulator::MultififoRegionizerEmulator(unsigned int nendcaps,
                                                               unsigned int nclocks,
                                                               unsigned int ntklinks,
                                                               unsigned int ncalolinks,
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
      NCALO_LINKS(ncalolinks),
      HCAL_LINKS(0),
      ECAL_LINKS(0),
      NMU_LINKS(1),
      nendcaps_(nendcaps),
      nclocks_(nclocks),
      ntk_(ntk),
      ncalo_(ncalo),
      nem_(nem),
      nmu_(nmu),
      outii_(outii),
      pauseii_(pauseii),
      streaming_(streaming),
      emInterceptMode_(noIntercept),
      init_(false),
      tkRegionizer_(ntk, streaming ? (ntk + outii - 1) / outii : ntk, streaming, outii, pauseii, useAlsoVtxCoords),
      hadCaloRegionizer_(ncalo, streaming ? (ncalo + outii - 1) / outii : ncalo, streaming, outii, pauseii),
      emCaloRegionizer_(nem, streaming ? (nem + outii - 1) / outii : nem, streaming, outii, pauseii),
      muRegionizer_(nmu, streaming ? std::max(1u, (nmu + outii - 1) / outii) : nmu, streaming, outii, pauseii) {
  // now we initialize the routes: track finder
  for (unsigned int ie = 0; ie < nendcaps && ntk > 0; ++ie) {
    for (unsigned int is = 0; is < NTK_SECTORS; ++is) {  // 9 tf sectors
      for (unsigned int il = 0; il < NTK_LINKS; ++il) {  // max tracks per sector per clock
        unsigned int isp = (is + 1) % NTK_SECTORS, ism = (is + NTK_SECTORS - 1) % NTK_SECTORS;
        tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, is + NTK_SECTORS * ie, il);
        tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, isp + NTK_SECTORS * ie, il + NTK_LINKS);
        tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, ism + NTK_SECTORS * ie, il + 2 * NTK_LINKS);
      }
    }
  }
  // hgcal
  assert(NCALO_SECTORS == 3 && NTK_SECTORS == 9);  // otherwise math below is broken, but it's hard to make it generic
  for (unsigned int ie = 0; ie < nendcaps; ++ie) {
    for (unsigned int is = 0; is < NCALO_SECTORS; ++is) {                      // NCALO_SECTORS sectors
      for (unsigned int il = 0; il < NCALO_LINKS; ++il) {                      // max clusters per sector per clock
        for (unsigned int j = 0; j < 3; ++j) {                                 // PF REGION
          caloRoutes_.emplace_back(is + 3 * ie, il, 3 * is + j + 9 * ie, il);  // 4 args are: sector, link, region, fifo
          if (j != 2) {  // pf regions 0 and 1 take also from previous sector
            int isprev = (is > 0 ? is - 1 : NCALO_SECTORS - 1);
            caloRoutes_.emplace_back(isprev + 3 * ie, il, 3 * is + j + 9 * ie, il + NCALO_LINKS);
          }
        }
      }
    }
  }
  emCaloRoutes_ = caloRoutes_;  // in the endcaps there's only one calo
  // mu
  for (unsigned int il = 0; il < NMU_LINKS && nmu > 0; ++il) {  // max clusters per sector per clock
    for (unsigned int j = 0; j < NTK_SECTORS * nendcaps; ++j) {
      muRoutes_.emplace_back(0, il, j, il);
    }
  }
}

l1ct::MultififoRegionizerEmulator::MultififoRegionizerEmulator(BarrelSetup barrelSetup,
                                                               unsigned int nHCalLinks,
                                                               unsigned int nECalLinks,
                                                               unsigned int nclocks,
                                                               unsigned int ntk,
                                                               unsigned int ncalo,
                                                               unsigned int nem,
                                                               unsigned int nmu,
                                                               bool streaming,
                                                               unsigned int outii,
                                                               unsigned int pauseii,
                                                               bool useAlsoVtxCoords)
    : RegionizerEmulator(useAlsoVtxCoords),
      NTK_SECTORS((barrelSetup == BarrelSetup::Phi18 || barrelSetup == BarrelSetup::Phi9) ? 5 : 9),
      NCALO_SECTORS((barrelSetup == BarrelSetup::Phi18 || barrelSetup == BarrelSetup::Phi9) ? 2 : 3),
      NTK_LINKS(2),
      NCALO_LINKS(2),
      HCAL_LINKS(nHCalLinks),
      ECAL_LINKS(nECalLinks),
      NMU_LINKS(1),
      nendcaps_(0),
      nclocks_(nclocks),
      ntk_(ntk),
      ncalo_(ncalo),
      nem_(nem),
      nmu_(nmu),
      outii_(outii),
      pauseii_(pauseii),
      streaming_(streaming),
      emInterceptMode_(noIntercept),
      init_(false),
      tkRegionizer_(ntk, streaming ? (ntk + outii - 1) / outii : ntk, streaming, outii, pauseii, useAlsoVtxCoords),
      hadCaloRegionizer_(ncalo, streaming ? (ncalo + outii - 1) / outii : ncalo, streaming, outii, pauseii),
      emCaloRegionizer_(nem, streaming ? (nem + outii - 1) / outii : nem, streaming, outii, pauseii),
      muRegionizer_(nmu, streaming ? std::max(1u, (nmu + outii - 1) / outii) : nmu, streaming, outii, pauseii) {
  unsigned int nendcaps = 2, etaslices = 0;
  switch (barrelSetup) {
    case BarrelSetup::Full54:
      nregions_ = 54;
      etaslices = 6;
      break;
    case BarrelSetup::Full27:
      nregions_ = 27;
      etaslices = 3;
      break;
    case BarrelSetup::Central18:
      nregions_ = 18;
      etaslices = 2;
      break;
    case BarrelSetup::Central9:
      nregions_ = 9;
      etaslices = 1;
      break;
    case BarrelSetup::Phi18:
      nregions_ = 18;
      etaslices = 6;
      break;
    case BarrelSetup::Phi9:
      nregions_ = 9;
      etaslices = 3;
      break;
  }
  unsigned int phisectors = nregions_ / etaslices;
  // now we initialize the routes: track finder
  for (unsigned int ietaslice = 0; ietaslice < etaslices && ntk > 0; ++ietaslice) {
    for (unsigned int ie = 0; ie < nendcaps; ++ie) {  // 0 = negative, 1 = positive
      unsigned int nTFEtaSlices = 1;
      if (etaslices == 3) {
        if (ietaslice == 0 && ie == 1)
          continue;
        if (ietaslice == 2 && ie == 0)
          continue;
        if (ietaslice == 1)
          nTFEtaSlices = 2;
      } else if (etaslices == 6) {
        if (ietaslice <= 1 && ie == 1)
          continue;
        if (ietaslice >= 4 && ie == 0)
          continue;
        if (ietaslice == 2 || ietaslice == 3)
          nTFEtaSlices = 2;
      } else if (barrelSetup == BarrelSetup::Central18 || barrelSetup == BarrelSetup::Central9) {
        nTFEtaSlices = 2;
      }
      unsigned int ireg0 = phisectors * ietaslice, il0 = 6 * (nTFEtaSlices - 1) * ie;
      if (barrelSetup == BarrelSetup::Phi18 || barrelSetup == BarrelSetup::Phi9) {
        for (unsigned int iregphi = 0; iregphi < (nregions_ / etaslices); ++iregphi) {
          for (unsigned int il = 0; il < NTK_LINKS; ++il) {
            tkRoutes_.emplace_back((iregphi + 1) + NTK_SECTORS * ie, il, iregphi + ireg0, il0 + il);
            tkRoutes_.emplace_back((iregphi + 0) + NTK_SECTORS * ie, il, iregphi + ireg0, il0 + il + 2);
            tkRoutes_.emplace_back((iregphi + 2) + NTK_SECTORS * ie, il, iregphi + ireg0, il0 + il + 4);
          }
        }
      } else {
        for (unsigned int is = 0; is < NTK_SECTORS; ++is) {  // 9 tf sectors
          for (unsigned int il = 0; il < NTK_LINKS; ++il) {  // max tracks per sector per clock
            unsigned int isp = (is + 1) % NTK_SECTORS, ism = (is + NTK_SECTORS - 1) % NTK_SECTORS;
            tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, is + ireg0, il0 + il);
            tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, isp + ireg0, il0 + il + 2);
            tkRoutes_.emplace_back(is + NTK_SECTORS * ie, il, ism + ireg0, il0 + il + 4);
          }
        }
      }
    }
  }
  // calo
  unsigned int calo_sectors_to_loop = NCALO_SECTORS;
  if (barrelSetup == BarrelSetup::Phi18 || barrelSetup == BarrelSetup::Phi9) {
    calo_sectors_to_loop = 1;
    assert(NCALO_SECTORS == 2 && NTK_SECTORS == 5);  // otherwise math below is broken, but it's hard to make it generic
  } else {
    assert(NCALO_SECTORS == 3 && NTK_SECTORS == 9);  // otherwise math below is broken, but it's hard to make it generic
  }
  for (unsigned int ie = 0; ie < etaslices; ++ie) {
    for (unsigned int is = 0; is < calo_sectors_to_loop; ++is) {  // NCALO_SECTORS sectors
      for (unsigned int j = 0; j < 3; ++j) {                      // 3 regions x sector
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
    for (unsigned int j = 0; j < nregions_; ++j) {
      muRoutes_.emplace_back(0, il, j, il);
    }
  }
}

l1ct::MultififoRegionizerEmulator::~MultififoRegionizerEmulator() {}

l1ct::MultififoRegionizerEmulator::BarrelSetup l1ct::MultififoRegionizerEmulator::parseBarrelSetup(
    const std::string& setup) {
  if (setup == "Full54")
    return BarrelSetup::Full54;
  if (setup == "Full27")
    return BarrelSetup::Full27;
  throw std::invalid_argument("barrelSetup for CMSSW can only be Full54 or Full27");
  return BarrelSetup::Full54;
}

void l1ct::MultififoRegionizerEmulator::setEgInterceptMode(bool afterFifo,
                                                           const l1ct::EGInputSelectorEmuConfig& interceptorConfig) {
  emInterceptMode_ = afterFifo ? interceptPostFifo : interceptPreFifo;
  interceptor_ = std::make_unique<EGInputSelectorEmulator>(interceptorConfig);
}

void l1ct::MultififoRegionizerEmulator::initSectorsAndRegions(const RegionizerDecodedInputs& in,
                                                              const std::vector<PFInputRegion>& out) {
  assert(!init_);
  init_ = true;
  if (nendcaps_ > 0) {
    assert(out.size() == NTK_SECTORS * nendcaps_);
  } else {
    assert(out.size() == nregions_);
  }
  nregions_ = out.size();
  if (ntk_) {
    assert(in.track.size() == NTK_SECTORS * (nendcaps_ ? nendcaps_ : 2));
    tkRegionizer_.initSectors(in.track);
    tkRegionizer_.initRegions(out);
    tkRegionizer_.initRouting(tkRoutes_);
  }
  if (ncalo_) {
    assert(in.hadcalo.size() == NCALO_SECTORS * (nendcaps_ ? nendcaps_ : 1));
    hadCaloRegionizer_.initSectors(in.hadcalo);
    hadCaloRegionizer_.initRegions(out);
    hadCaloRegionizer_.initRouting(caloRoutes_);
  }
  if (nem_) {
    assert(in.emcalo.size() == NCALO_SECTORS * (nendcaps_ ? nendcaps_ : 1));
    emCaloRegionizer_.initSectors(in.emcalo);
    emCaloRegionizer_.initRegions(out);
    emCaloRegionizer_.initRouting(emCaloRoutes_);
  }
  if (nmu_) {
    muRegionizer_.initSectors(in.muon);
    muRegionizer_.initRegions(out);
    muRegionizer_.initRouting(muRoutes_);
  }
}

// clock-cycle emulation
bool l1ct::MultififoRegionizerEmulator::step(bool newEvent,
                                             const std::vector<l1ct::TkObjEmu>& links,
                                             std::vector<l1ct::TkObjEmu>& out,
                                             bool mux) {
  return ntk_ ? tkRegionizer_.step(newEvent, links, out, mux) : false;
}

bool l1ct::MultififoRegionizerEmulator::step(bool newEvent,
                                             const std::vector<l1ct::EmCaloObjEmu>& links,
                                             std::vector<l1ct::EmCaloObjEmu>& out,
                                             bool mux) {
  assert(emInterceptMode_ == noIntercept);  // otherwise the em & had calo can't be stepped independently
  return nem_ ? emCaloRegionizer_.step(newEvent, links, out, mux) : false;
}

bool l1ct::MultififoRegionizerEmulator::step(bool newEvent,
                                             const std::vector<l1ct::HadCaloObjEmu>& links,
                                             std::vector<l1ct::HadCaloObjEmu>& out,
                                             bool mux) {
  return ncalo_ ? hadCaloRegionizer_.step(newEvent, links, out, mux) : false;
}

bool l1ct::MultififoRegionizerEmulator::step(bool newEvent,
                                             const std::vector<l1ct::MuObjEmu>& links,
                                             std::vector<l1ct::MuObjEmu>& out,
                                             bool mux) {
  return nmu_ ? muRegionizer_.step(newEvent, links, out, mux) : false;
}

bool l1ct::MultififoRegionizerEmulator::step(bool newEvent,
                                             const std::vector<l1ct::TkObjEmu>& links_tk,
                                             const std::vector<l1ct::HadCaloObjEmu>& links_hadCalo,
                                             const std::vector<l1ct::EmCaloObjEmu>& links_emCalo,
                                             const std::vector<l1ct::MuObjEmu>& links_mu,
                                             std::vector<l1ct::TkObjEmu>& out_tk,
                                             std::vector<l1ct::HadCaloObjEmu>& out_hadCalo,
                                             std::vector<l1ct::EmCaloObjEmu>& out_emCalo,
                                             std::vector<l1ct::MuObjEmu>& out_mu,
                                             bool mux) {
  bool ret = false;
  if (ntk_)
    ret = tkRegionizer_.step(newEvent, links_tk, out_tk, mux);
  if (nmu_)
    ret = muRegionizer_.step(newEvent, links_mu, out_mu, mux);
  switch (emInterceptMode_) {
    case noIntercept:
      if (ncalo_)
        ret = hadCaloRegionizer_.step(newEvent, links_hadCalo, out_hadCalo, mux);
      if (nem_)
        ret = emCaloRegionizer_.step(newEvent, links_emCalo, out_emCalo, mux);
      break;
    case interceptPreFifo:
      // we actually intercept at the links, in the software it's equivalent and it's easier
      assert(nem_ > 0 && ncalo_ > 0 && !links_hadCalo.empty() && links_emCalo.empty());
      assert(interceptor_.get());
      {
        std::vector<l1ct::EmCaloObjEmu> intercepted_links;
        interceptor_->select_or_clear(links_hadCalo, intercepted_links);
        ret = hadCaloRegionizer_.step(newEvent, links_hadCalo, out_hadCalo, mux);
        emCaloRegionizer_.step(newEvent, intercepted_links, out_emCalo, mux);
      }
      break;
    case interceptPostFifo:
      assert(nem_ > 0 && ncalo_ > 0 && !links_hadCalo.empty() && links_emCalo.empty());
      assert(interceptor_.get());
      {
        if (mux) {
          std::vector<l1ct::HadCaloObjEmu> hadNoMux;
          hadCaloRegionizer_.step(newEvent, links_hadCalo, hadNoMux, /*mux=*/false);
          std::vector<l1ct::EmCaloObjEmu> emNoMux(hadNoMux.size());
          interceptor_->select_or_clear(hadNoMux, emNoMux);
          ret = hadCaloRegionizer_.muxonly_step(newEvent, /*flush=*/false, hadNoMux, out_hadCalo);
          emCaloRegionizer_.muxonly_step(newEvent, /*flush=*/true, emNoMux, out_emCalo);
        } else {
          ret = hadCaloRegionizer_.step(newEvent, links_hadCalo, out_hadCalo, /*mux=*/false);
          interceptor_->select_or_clear(out_hadCalo, out_emCalo);
        }
      }
      break;
  }
  return ret;
}

void l1ct::MultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                  const l1ct::RegionizerDecodedInputs& in,
                                                  std::vector<l1ct::TkObjEmu>& links,
                                                  std::vector<bool>& valid) {
  if (ntk_ == 0)
    return;
  links.resize(NTK_SECTORS * NTK_LINKS * (nendcaps_ ? nendcaps_ : 2));
  valid.resize(links.size());
  for (unsigned int is = 0, idx = 0; is < NTK_SECTORS * (nendcaps_ ? nendcaps_ : 2); ++is) {  // tf sectors
    const l1ct::DetectorSector<l1ct::TkObjEmu>& sec = in.track[is];
    for (unsigned int il = 0; il < NTK_LINKS; ++il, ++idx) {
      unsigned int ioffs = iclock * NTK_LINKS + il;
      if (ioffs < sec.size() && iclock < nclocks_ - 1) {
        links[idx] = sec[ioffs];
        valid[idx] = true;
      } else {
        links[idx].clear();
        valid[idx] = false;
      }
    }
  }
}

template <typename T>
void l1ct::MultififoRegionizerEmulator::fillCaloLinks(unsigned int iclock,
                                                      const std::vector<DetectorSector<T>>& in,
                                                      std::vector<T>& links,
                                                      std::vector<bool>& valid) {
  unsigned int NLINKS =
      (nendcaps_ ? NCALO_LINKS : (typeid(T) == typeid(l1ct::HadCaloObjEmu) ? HCAL_LINKS : ECAL_LINKS));
  links.resize(NCALO_SECTORS * (nendcaps_ ? nendcaps_ : 1) * NLINKS);
  valid.resize(links.size());
  for (unsigned int is = 0, idx = 0; is < NCALO_SECTORS * (nendcaps_ ? nendcaps_ : 1); ++is) {
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

void l1ct::MultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                  const l1ct::RegionizerDecodedInputs& in,
                                                  std::vector<l1ct::HadCaloObjEmu>& links,
                                                  std::vector<bool>& valid) {
  if (ncalo_ == 0)
    return;
  fillCaloLinks(iclock, in.hadcalo, links, valid);
}

void l1ct::MultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                  const l1ct::RegionizerDecodedInputs& in,
                                                  std::vector<l1ct::EmCaloObjEmu>& links,
                                                  std::vector<bool>& valid) {
  if (nem_ == 0 || emInterceptMode_ != noIntercept)
    return;
  fillCaloLinks(iclock, in.emcalo, links, valid);
}

void l1ct::MultififoRegionizerEmulator::fillLinks(unsigned int iclock,
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

void l1ct::MultififoRegionizerEmulator::toFirmware(const std::vector<l1ct::TkObjEmu>& emu,
                                                   TkObj fw[/*NTK_SECTORS][NTK_LINKS*/]) {
  if (ntk_ == 0)
    return;
  assert(emu.size() == NTK_SECTORS * NTK_LINKS * (nendcaps_ ? nendcaps_ : 2));
  for (unsigned int is = 0, idx = 0; is < NTK_SECTORS * (nendcaps_ ? nendcaps_ : 2); ++is) {  // tf sectors
    for (unsigned int il = 0; il < NTK_LINKS; ++il, ++idx) {
      fw[is * NTK_LINKS + il] = emu[idx];
    }
  }
}
void l1ct::MultififoRegionizerEmulator::toFirmware(const std::vector<l1ct::HadCaloObjEmu>& emu,
                                                   HadCaloObj fw[/*NCALO_SECTORS*NCALO_LINKS*/]) {
  if (ncalo_ == 0)
    return;
  unsigned int NLINKS = (nendcaps_ ? NCALO_LINKS * nendcaps_ : HCAL_LINKS);
  assert(emu.size() == NCALO_SECTORS * NLINKS);
  for (unsigned int is = 0, idx = 0; is < NCALO_SECTORS * (nendcaps_ ? nendcaps_ : 1); ++is) {  // calo sectors
    for (unsigned int il = 0; il < NLINKS; ++il, ++idx) {
      fw[is * NLINKS + il] = emu[idx];
    }
  }
}

void l1ct::MultififoRegionizerEmulator::toFirmware(const std::vector<l1ct::EmCaloObjEmu>& emu,
                                                   EmCaloObj fw[/*NCALO_SECTORS*NCALO_LINKS*/]) {
  if (nem_ == 0)
    return;
  unsigned int NLINKS = (nendcaps_ ? NCALO_LINKS * nendcaps_ : ECAL_LINKS);
  assert(emu.size() == NCALO_SECTORS * NLINKS);
  for (unsigned int is = 0, idx = 0; is < NCALO_SECTORS * (nendcaps_ ? nendcaps_ : 1); ++is) {  // calo sectors
    for (unsigned int il = 0; il < NLINKS; ++il, ++idx) {
      fw[is * NLINKS + il] = emu[idx];
    }
  }
}

void l1ct::MultififoRegionizerEmulator::toFirmware(const std::vector<l1ct::MuObjEmu>& emu, MuObj fw[/*NMU_LINKS*/]) {
  if (nmu_ == 0)
    return;
  assert(emu.size() == NMU_LINKS);
  for (unsigned int il = 0, idx = 0; il < NMU_LINKS; ++il, ++idx) {
    fw[il] = emu[idx];
  }
}

void l1ct::MultififoRegionizerEmulator::destream(int iclock,
                                                 const std::vector<l1ct::TkObjEmu>& tk_out,
                                                 const std::vector<l1ct::EmCaloObjEmu>& em_out,
                                                 const std::vector<l1ct::HadCaloObjEmu>& calo_out,
                                                 const std::vector<l1ct::MuObjEmu>& mu_out,
                                                 PFInputRegion& out) {
  if (ntk_)
    tkRegionizer_.destream(iclock, tk_out, out.track);
  if (ncalo_)
    hadCaloRegionizer_.destream(iclock, calo_out, out.hadcalo);
  if (nem_)
    emCaloRegionizer_.destream(iclock, em_out, out.emcalo);
  if (nmu_)
    muRegionizer_.destream(iclock, mu_out, out.muon);
}

void l1ct::MultififoRegionizerEmulator::reset() {
  tkRegionizer_.reset();
  emCaloRegionizer_.reset();
  hadCaloRegionizer_.reset();
  muRegionizer_.reset();
}

void l1ct::MultififoRegionizerEmulator::run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) {
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

    bool newevt = (iclock == 0), mux = true;
    step(newevt, tk_links_in, calo_links_in, em_links_in, mu_links_in, tk_out, calo_out, em_out, mu_out, mux);
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
  assert(out.size() == nregions_);
  for (unsigned int iclock = 0; iclock < nclocks_; ++iclock) {
    bool newevt = (iclock == 0), mux = true;
    step(newevt, tk_links_in, calo_links_in, em_links_in, mu_links_in, tk_out, calo_out, em_out, mu_out, mux);

    unsigned int ireg = iclock / (outii_ + pauseii_);
    if ((iclock % (outii_ + pauseii_)) >= outii_)
      continue;
    if (ireg >= nregions_)
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
