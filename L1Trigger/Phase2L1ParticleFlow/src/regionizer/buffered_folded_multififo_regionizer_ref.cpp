#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/buffered_folded_multififo_regionizer_ref.h"

#include <iostream>
#include <memory>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::BufferedFoldedMultififoRegionizerEmulator::BufferedFoldedMultififoRegionizerEmulator(
    const edm::ParameterSet& iConfig)
    : BufferedFoldedMultififoRegionizerEmulator(iConfig.getParameter<uint32_t>("nClocks"),
                                                iConfig.getParameter<uint32_t>("nTrack"),
                                                iConfig.getParameter<uint32_t>("nCalo"),
                                                iConfig.getParameter<uint32_t>("nEmCalo"),
                                                iConfig.getParameter<uint32_t>("nMu"),
                                                /*streaming=*/true,
                                                /*outii=*/6,
                                                /*pauseii=*/3,
                                                iConfig.getParameter<bool>("useAlsoVtxCoords")) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug");
  if (iConfig.existsAs<edm::ParameterSet>("egInterceptMode")) {
    const auto& emSelCfg = iConfig.getParameter<edm::ParameterSet>("egInterceptMode");
    setEgInterceptMode(emSelCfg.getParameter<bool>("afterFifo"), emSelCfg);
  }
}

edm::ParameterSetDescription l1ct::BufferedFoldedMultififoRegionizerEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<uint32_t>("nClocks", 162);
  description.add<uint32_t>("nTrack", 30);
  description.add<uint32_t>("nCalo", 20);
  description.add<uint32_t>("nEmCalo", 10);
  description.add<uint32_t>("nMu", 4);
  edm::ParameterSetDescription egIntercept = l1ct::EGInputSelectorEmuConfig::getParameterSetDescription();
  egIntercept.add<bool>("afterFifo", true);
  description.addOptional<edm::ParameterSetDescription>("egInterceptMode", egIntercept);
  description.add<bool>("useAlsoVtxCoords", true);
  description.addUntracked<bool>("debug", false);
  return description;
}
#endif

l1ct::BufferedFoldedMultififoRegionizerEmulator::BufferedFoldedMultififoRegionizerEmulator(unsigned int nclocks,
                                                                                           unsigned int ntk,
                                                                                           unsigned int ncalo,
                                                                                           unsigned int nem,
                                                                                           unsigned int nmu,
                                                                                           bool streaming,
                                                                                           unsigned int outii,
                                                                                           unsigned int pauseii,
                                                                                           bool useAlsoVtxCoords)
    : FoldedMultififoRegionizerEmulator(nclocks,
                                        /*NTK_LINKS*/ 1,
                                        /*NCALO_LINKS*/ 1,
                                        ntk,
                                        ncalo,
                                        nem,
                                        nmu,
                                        streaming,
                                        outii,
                                        pauseii,
                                        useAlsoVtxCoords),
      tkBuffers_(ntk ? 2 * NTK_SECTORS : 0),
      caloBuffers_(ncalo ? 2 * NCALO_SECTORS : 0),
      muBuffers_(nmu ? 2 : 0) {}

l1ct::BufferedFoldedMultififoRegionizerEmulator::~BufferedFoldedMultififoRegionizerEmulator() {}

void l1ct::BufferedFoldedMultififoRegionizerEmulator::findEtaBounds_(const l1ct::PFRegionEmu& sec,
                                                                     const std::vector<PFInputRegion>& reg,
                                                                     l1ct::glbeta_t& etaMin,
                                                                     l1ct::glbeta_t& etaMax) {
  etaMin = reg[0].region.hwEtaCenter - reg[0].region.hwEtaHalfWidth - reg[0].region.hwEtaExtra - sec.hwEtaCenter;
  etaMax = reg[0].region.hwEtaCenter + reg[0].region.hwEtaHalfWidth + reg[0].region.hwEtaExtra - sec.hwEtaCenter;
  for (const auto& r : reg) {
    etaMin = std::min<l1ct::glbeta_t>(
        etaMin, r.region.hwEtaCenter - r.region.hwEtaHalfWidth - r.region.hwEtaExtra - sec.hwEtaCenter);
    etaMax = std::max<l1ct::glbeta_t>(
        etaMax, r.region.hwEtaCenter + r.region.hwEtaHalfWidth + r.region.hwEtaExtra - sec.hwEtaCenter);
  }
}
void l1ct::BufferedFoldedMultififoRegionizerEmulator::initSectorsAndRegions(const RegionizerDecodedInputs& in,
                                                                            const std::vector<PFInputRegion>& out) {
  assert(!init_);
  FoldedMultififoRegionizerEmulator::initSectorsAndRegions(in, out);
  for (int ie = 0; ie < 2; ++ie) {
    l1ct::glbeta_t etaMin, etaMax;
    findEtaBounds_(fold_[ie].sectors.track[0].region, fold_[ie].regions, etaMin, etaMax);
    for (unsigned int isec = 0; ntk_ > 0 && isec < NTK_SECTORS; ++isec) {
      tkBuffers_[2 * isec + ie] = l1ct::multififo_regionizer::EtaBuffer<l1ct::TkObjEmu>(nclocks_ / 2, etaMin, etaMax);
    }
    findEtaBounds_(fold_[ie].sectors.hadcalo[0].region, fold_[ie].regions, etaMin, etaMax);
    for (unsigned int isec = 0; ncalo_ > 0 && isec < NCALO_SECTORS; ++isec) {
      caloBuffers_[2 * isec + ie] =
          l1ct::multififo_regionizer::EtaBuffer<l1ct::HadCaloObjEmu>(nclocks_ / 2, etaMin, etaMax);
    }
    findEtaBounds_(fold_[ie].sectors.muon.region, fold_[ie].regions, etaMin, etaMax);
    if (nmu_ > 0) {
      muBuffers_[ie] = l1ct::multififo_regionizer::EtaBuffer<l1ct::MuObjEmu>(nclocks_ / 2, etaMin, etaMax);
    }
  }
}

template <typename T>
void l1ct::BufferedFoldedMultififoRegionizerEmulator::fillLinksPosNeg_(
    unsigned int iclock,
    const std::vector<l1ct::DetectorSector<T>>& secNeg,
    const std::vector<l1ct::DetectorSector<T>>& secPos,
    std::vector<T>& links,
    std::vector<bool>& valid) {
  unsigned int nlinks = secNeg.size();
  links.resize(2 * nlinks);
  valid.resize(2 * nlinks);
  for (unsigned int isec = 0, ilink = 0; isec < nlinks; ++isec) {
    for (int ec = 0; ec < 2; ++ec, ++ilink) {
      const l1ct::DetectorSector<T>& sec = (ec ? secPos : secNeg)[isec];
      if (iclock < sec.size() && iclock < nclocks_ - 1) {
        valid[ilink] = true;
        links[ilink] = sec.obj[iclock];
      } else {
        valid[ilink] = false;
        links[ilink].clear();
      }
    }
  }
}
void l1ct::BufferedFoldedMultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                                std::vector<l1ct::TkObjEmu>& links,
                                                                std::vector<bool>& valid) {
  if (ntk_)
    fillLinksPosNeg_(iclock, fold_[0].sectors.track, fold_[1].sectors.track, links, valid);
}

void l1ct::BufferedFoldedMultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                                std::vector<l1ct::HadCaloObjEmu>& links,
                                                                std::vector<bool>& valid) {
  if (ncalo_)
    fillLinksPosNeg_(iclock, fold_[0].sectors.hadcalo, fold_[1].sectors.hadcalo, links, valid);
}
void l1ct::BufferedFoldedMultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                                std::vector<l1ct::EmCaloObjEmu>& links,
                                                                std::vector<bool>& valid) {
  // nothing to do normally
}

void l1ct::BufferedFoldedMultififoRegionizerEmulator::fillLinks(unsigned int iclock,
                                                                std::vector<l1ct::MuObjEmu>& links,
                                                                std::vector<bool>& valid) {
  if (nmu_ == 0)
    return;
  assert(NMU_LINKS == 1);
  links.resize(NMU_LINKS);
  valid.resize(links.size());
  const auto& in = fold_.front().sectors.muon.obj;
  if (iclock < in.size() && iclock < nclocks_ - 1) {
    links[0] = in[iclock];
    valid[0] = true;
  } else {
    links[0].clear();
    valid[0] = false;
  }
}

bool l1ct::BufferedFoldedMultififoRegionizerEmulator::step(bool newEvent,
                                                           const std::vector<l1ct::TkObjEmu>& links_tk,
                                                           const std::vector<l1ct::HadCaloObjEmu>& links_hadCalo,
                                                           const std::vector<l1ct::EmCaloObjEmu>& links_emCalo,
                                                           const std::vector<l1ct::MuObjEmu>& links_mu,
                                                           std::vector<l1ct::TkObjEmu>& out_tk,
                                                           std::vector<l1ct::HadCaloObjEmu>& out_hadCalo,
                                                           std::vector<l1ct::EmCaloObjEmu>& out_emCalo,
                                                           std::vector<l1ct::MuObjEmu>& out_mu,
                                                           bool mux) {
  iclock_ = (newEvent ? 0 : iclock_ + 1);
  int ifold = (iclock_ / clocksPerFold_);
  bool newSubEvent = iclock_ % clocksPerFold_ == 0;

  if (newEvent) {
    for (auto& b : tkBuffers_)
      b.writeNewEvent();
    for (auto& b : caloBuffers_)
      b.writeNewEvent();
    for (auto& b : muBuffers_)
      b.writeNewEvent();
  }

  assert(links_tk.size() == tkBuffers_.size() || ntk_ == 0);
  for (unsigned int i = 0; ntk_ > 0 && i < 2 * NTK_SECTORS; ++i) {
    tkBuffers_[i].maybe_push(links_tk[i]);
  }
  assert(links_hadCalo.size() == caloBuffers_.size() || ncalo_ == 0);
  for (unsigned int i = 0; ncalo_ > 0 && i < 2 * NCALO_SECTORS; ++i) {
    caloBuffers_[i].maybe_push(links_hadCalo[i]);
  }
  for (unsigned int i = 0; nmu_ > 0 && i < 2; ++i) {
    muBuffers_[i].maybe_push(links_mu[0]);
  }
  if (newSubEvent && !newEvent) {
    for (auto& b : tkBuffers_)
      b.readNewEvent();
    for (auto& b : caloBuffers_)
      b.readNewEvent();
    for (auto& b : muBuffers_)
      b.readNewEvent();
  }
  std::vector<l1ct::TkObjEmu> mylinks_tk(ntk_ ? NTK_SECTORS : 0);
  std::vector<l1ct::HadCaloObjEmu> mylinks_hadCalo(ncalo_ ? NCALO_SECTORS : 0);
  std::vector<l1ct::EmCaloObjEmu> mylinks_emCalo(0);
  std::vector<l1ct::MuObjEmu> mylinks_mu(nmu_ ? 1 : 0);
  for (unsigned int i = 0, ib = 1 - ifold; ntk_ > 0 && i < NTK_SECTORS; ++i, ib += 2) {
    mylinks_tk[i] = tkBuffers_[ib].pop();
  }
  for (unsigned int i = 0, ib = 1 - ifold; ncalo_ > 0 && i < NCALO_SECTORS; ++i, ib += 2) {
    mylinks_hadCalo[i] = caloBuffers_[ib].pop();
  }
  if (nmu_) {
    mylinks_mu[0] = muBuffers_[1 - ifold].pop();
  }

  bool ret = false;
  if (!mux) {
    Fold& f = fold_[1 - ifold];
    ret = f.regionizer->step(newSubEvent,
                             mylinks_tk,
                             mylinks_hadCalo,
                             mylinks_emCalo,
                             mylinks_mu,
                             out_tk,
                             out_hadCalo,
                             out_emCalo,
                             out_mu,
                             false);
  } else {
    unsigned int inputFold = 1 - ifold;
    unsigned int outputFold = ifold;
    std::vector<l1ct::TkObjEmu> nolinks_tk(mylinks_tk.size());
    std::vector<l1ct::HadCaloObjEmu> nolinks_hadCalo(mylinks_hadCalo.size());
    std::vector<l1ct::EmCaloObjEmu> nolinks_emCalo(mylinks_emCalo.size());
    std::vector<l1ct::MuObjEmu> nolinks_mu(mylinks_mu.size());
    std::vector<l1ct::TkObjEmu> noout_tk;
    std::vector<l1ct::HadCaloObjEmu> noout_hadCalo;
    std::vector<l1ct::EmCaloObjEmu> noout_emCalo;
    std::vector<l1ct::MuObjEmu> noout_mu;
    for (auto& f : fold_) {
      bool fret = f.regionizer->step(newSubEvent,
                                     f.index == inputFold ? mylinks_tk : nolinks_tk,
                                     f.index == inputFold ? mylinks_hadCalo : nolinks_hadCalo,
                                     f.index == inputFold ? mylinks_emCalo : nolinks_emCalo,
                                     f.index == inputFold ? mylinks_mu : nolinks_mu,
                                     f.index == outputFold ? out_tk : noout_tk,
                                     f.index == outputFold ? out_hadCalo : noout_hadCalo,
                                     f.index == outputFold ? out_emCalo : noout_emCalo,
                                     f.index == outputFold ? out_mu : noout_mu,
                                     true);
      if (f.index == outputFold)
        ret = fret;
    }
  }
  return ret;
}

void l1ct::BufferedFoldedMultififoRegionizerEmulator::run(const RegionizerDecodedInputs& in,
                                                          std::vector<PFInputRegion>& out) {
  if (!init_)
    initSectorsAndRegions(in, out);
  else {
    fillEvent(in);
    for (auto& f : fold_)
      f.regionizer->reset();
  }

  std::vector<l1ct::TkObjEmu> tk_links_in, tk_out;
  std::vector<l1ct::EmCaloObjEmu> em_links_in, em_out;
  std::vector<l1ct::HadCaloObjEmu> calo_links_in, calo_out;
  std::vector<l1ct::MuObjEmu> mu_links_in, mu_out;

  // read and sort the inputs
  std::vector<bool> unused;
  for (unsigned int iclock = 0; iclock < nclocks_; ++iclock) {
    fillLinks(iclock, tk_links_in, unused);
    fillLinks(iclock, em_links_in, unused);
    fillLinks(iclock, calo_links_in, unused);
    fillLinks(iclock, mu_links_in, unused);

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
      Fold& f = fold_[whichFold(iclock)];
      f.regionizer->destream(iclock % clocksPerFold_, tk_out, em_out, calo_out, mu_out, out[ireg]);
    } else {
      if (iclock % (outii_ + pauseii_) == 0) {
        out[ireg].track = tk_out;
        out[ireg].emcalo = em_out;
        out[ireg].hadcalo = calo_out;
        out[ireg].muon = mu_out;
      }
    }
  }

  for (auto& f : fold_)
    f.regionizer->reset();
}
