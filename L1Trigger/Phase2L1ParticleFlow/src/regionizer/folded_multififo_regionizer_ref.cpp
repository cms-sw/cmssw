#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/folded_multififo_regionizer_ref.h"

#include <iostream>
#include <memory>

l1ct::FoldedMultififoRegionizerEmulator::FoldedMultififoRegionizerEmulator(unsigned int nclocks,
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
      NTK_LINKS(2),
      NCALO_LINKS((outii + pauseii == 9 ? 2 : 3)),
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
    fold_.emplace_back(
        ie,
        std::make_unique<l1ct::MultififoRegionizerEmulator>(
            /*nendcaps=*/1, nclocks / 2, ntk, ncalo, nem, nmu, streaming, outii, pauseii, useAlsoVtxCoords));
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
  std::cout << "Initializing folded  with " << in.track.size() << " tk sectors, " << out.size() << " regions"
            << std::endl;
  for (auto& f : fold_) {
    std::cout << "Initializing fold " << f.index << " with " << f.sectors.track.size() << " tk sectors, "
              << f.regions.size() << " regions" << std::endl;
    f.regionizer->initSectorsAndRegions(f.sectors, f.regions);
  }
}
// clock-cycle emulation
#if 0
bool l1ct::FoldedMultififoRegionizerEmulator::step(bool newEvent,
                                                   const std::vector<l1ct::HadCaloObjEmu>& links,
                                                   std::vector<l1ct::HadCaloObjEmu>& out,
                                                   bool mux) {
}
#endif

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
  else
    fillEvent(in);
  for (auto& f : fold_)
    f.regionizer->run(f.sectors, f.regions);
  for (auto& o : out) {
    for (auto& ro : fold_[whichFold(o.region)].regions) {
      if (ro.region.hwEtaCenter == o.region.hwEtaCenter && ro.region.hwPhiCenter == o.region.hwPhiCenter) {
        std::swap(o.track, ro.track);
        std::swap(o.hadcalo, ro.hadcalo);
        std::swap(o.emcalo, ro.emcalo);
        std::swap(o.muon, ro.muon);
        break;
      }
    }
  }
}
