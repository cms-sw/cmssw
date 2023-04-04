#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/folded_multififo_regionizer_ref.h"

#include <iostream>
#include <memory>

l1ct::FoldedMultififoRegionizerEmulator::FoldedMultififoRegionizerEmulator(unsigned int nclocks,
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
      nclocks_(nclocks),
      ntk_(ntk),
      ncalo_(ncalo),
      nem_(nem),
      nmu_(nmu),
      outii_(outii),
      pauseii_(pauseii),
      streaming_(streaming),
      foldMode_(FoldMode::EndcapEta2),
      init_(false) {
  // now we initialize the routes: track finder
  for (unsigned int ie = 0; ie < 2; ++ie) {
    fold_.emplace_back(ie,
                       std::make_unique<l1ct::MultififoRegionizerEmulator>(
                           /*nendcaps=*/1,
                           nclocks / 2,
                           NTK_LINKS,
                           NCALO_LINKS,
                           ntk,
                           ncalo,
                           nem,
                           nmu,
                           streaming,
                           outii,
                           pauseii,
                           useAlsoVtxCoords));
  }
  clocksPerFold_ = nclocks / 2;
}

l1ct::FoldedMultififoRegionizerEmulator::~FoldedMultififoRegionizerEmulator() {}

void l1ct::FoldedMultififoRegionizerEmulator::setEgInterceptMode(
    bool afterFifo, const l1ct::EGInputSelectorEmuConfig& interceptorConfig) {
  for (auto& f : fold_)
    f.regionizer->setEgInterceptMode(afterFifo, interceptorConfig);
}

void l1ct::FoldedMultififoRegionizerEmulator::splitSectors(const RegionizerDecodedInputs& in) {
  for (auto& f : fold_) {
    f.sectors.track.clear();
    f.sectors.hadcalo.clear();
    f.sectors.emcalo.clear();
  }
  for (const auto& src : in.track) {
    for (auto& f : fold_) {
      if (inFold(src.region, f))
        f.sectors.track.emplace_back(src);
    }
  }
  for (const auto& src : in.hadcalo) {
    for (auto& f : fold_) {
      if (inFold(src.region, f))
        f.sectors.hadcalo.emplace_back(src);
    }
  }
  for (const auto& src : in.emcalo) {
    for (auto& f : fold_) {
      if (inFold(src.region, f))
        f.sectors.emcalo.emplace_back(src);
    }
  }
  for (auto& f : fold_)
    f.sectors.muon = in.muon;
}

void l1ct::FoldedMultififoRegionizerEmulator::splitRegions(const std::vector<PFInputRegion>& out) {
  for (auto& f : fold_) {
    f.regions.clear();
  }
  for (const auto& o : out) {
    fold_[whichFold(o.region)].regions.push_back(o);
  }
}

void l1ct::FoldedMultififoRegionizerEmulator::initSectorsAndRegions(const RegionizerDecodedInputs& in,
                                                                    const std::vector<PFInputRegion>& out) {
  assert(!init_);
  init_ = true;
  splitSectors(in);
  splitRegions(out);
  for (auto& f : fold_) {
    f.regionizer->initSectorsAndRegions(f.sectors, f.regions);
  }
  nregions_ = out.size();
}

bool l1ct::FoldedMultififoRegionizerEmulator::step(bool newEvent,
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
  bool newSubEvent = iclock_ % clocksPerFold_ == 0;
  bool ret = false;
  if (!mux) {
    Fold& f = fold_[whichFold(iclock_)];
    ret = f.regionizer->step(
        newSubEvent, links_tk, links_hadCalo, links_emCalo, links_mu, out_tk, out_hadCalo, out_emCalo, out_mu, false);
  } else {
    unsigned int inputFold = whichFold(iclock_);
    unsigned int outputFold = (inputFold + 1) % fold_.size();  // to be seen if this is general or not
    std::vector<l1ct::TkObjEmu> nolinks_tk(links_tk.size());
    std::vector<l1ct::HadCaloObjEmu> nolinks_hadCalo(links_hadCalo.size());
    std::vector<l1ct::EmCaloObjEmu> nolinks_emCalo(links_emCalo.size());
    std::vector<l1ct::MuObjEmu> nolinks_mu(links_mu.size());
    std::vector<l1ct::TkObjEmu> noout_tk;
    std::vector<l1ct::HadCaloObjEmu> noout_hadCalo;
    std::vector<l1ct::EmCaloObjEmu> noout_emCalo;
    std::vector<l1ct::MuObjEmu> noout_mu;
    for (auto& f : fold_) {
      bool fret = f.regionizer->step(newSubEvent,
                                     f.index == inputFold ? links_tk : nolinks_tk,
                                     f.index == inputFold ? links_hadCalo : nolinks_hadCalo,
                                     f.index == inputFold ? links_emCalo : nolinks_emCalo,
                                     f.index == inputFold ? links_mu : nolinks_mu,
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

void l1ct::FoldedMultififoRegionizerEmulator::fillEvent(const l1ct::RegionizerDecodedInputs& in) {
  splitSectors(in);
  for (auto& f : fold_) {
    for (auto& o : f.regions) {
      o.clear();
    }
  }
}

unsigned int l1ct::FoldedMultififoRegionizerEmulator::whichFold(const l1ct::PFRegion& reg) {
  switch (foldMode_) {
    case FoldMode::EndcapEta2:
      return (reg.floatEtaCenter() >= 0);
  }
  assert(false);
  return 0;
}
bool l1ct::FoldedMultififoRegionizerEmulator::inFold(const l1ct::PFRegion& reg, const Fold& fold) {
  switch (foldMode_) {
    case FoldMode::EndcapEta2:
      return int(reg.floatEtaCenter() >= 0) == int(fold.index);
  }
  assert(false);
  return false;
}

void l1ct::FoldedMultififoRegionizerEmulator::run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) {
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

  std::vector<bool> unused;
  for (unsigned int iclock = 0; iclock < 2 * nclocks_; ++iclock) {
    if (iclock < nclocks_) {
      fillLinks(iclock, tk_links_in, unused);
      fillLinks(iclock, em_links_in, unused);
      fillLinks(iclock, calo_links_in, unused);
      fillLinks(iclock, mu_links_in, unused);
    } else {
      // set up an empty event
      for (auto& l : tk_links_in)
        l.clear();
      for (auto& l : em_links_in)
        l.clear();
      for (auto& l : calo_links_in)
        l.clear();
      for (auto& l : mu_links_in)
        l.clear();
    }

    bool newevt = (iclock % nclocks_) == 0, mux = true;
    step(newevt, tk_links_in, calo_links_in, em_links_in, mu_links_in, tk_out, calo_out, em_out, mu_out, mux);

    if (iclock >= nclocks_ / 2) {
      unsigned int ireg = ((iclock - nclocks_ / 2) / (outii_ + pauseii_));
      if (((iclock - nclocks_ / 2) % (outii_ + pauseii_)) >= outii_)
        continue;
      if (ireg >= nregions_)
        break;

      if (streaming_) {
        Fold& f = fold_[whichFold(iclock)];
        f.regionizer->destream(iclock % clocksPerFold_, tk_out, em_out, calo_out, mu_out, out[ireg]);
      } else {
        if ((iclock - nclocks_ / 2) % (outii_ + pauseii_) == 0) {
          out[ireg].track = tk_out;
          out[ireg].emcalo = em_out;
          out[ireg].hadcalo = calo_out;
          out[ireg].muon = mu_out;
        }
      }
    }
  }

  for (auto& f : fold_)
    f.regionizer->reset();
}
