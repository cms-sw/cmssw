#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_alt_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_alt_regionizer_elements_ref.icc"

#include <iostream>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

l1ct::TDRAltRegionizerEmulator::TDRAltRegionizerEmulator(const edm::ParameterSet& iConfig)
    : TDRAltRegionizerEmulator(iConfig.getParameter<uint32_t>("nTrack"),
                               iConfig.getParameter<uint32_t>("nCalo"),
                               iConfig.getParameter<uint32_t>("nEmCalo"),
                               iConfig.getParameter<uint32_t>("nMu"),
                               iConfig.getUntrackedParameter<bool>("debug_tk"),
                               iConfig.getUntrackedParameter<bool>("debug_calo"),
                               iConfig.getUntrackedParameter<bool>("debug_emcalo"),
                               iConfig.getUntrackedParameter<bool>("debug_mu")
                              ) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug");
}

edm::ParameterSetDescription l1ct::TDRAltRegionizerEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<uint32_t>("nTrack", 22);
  description.add<uint32_t>("nCalo", 15);
  description.add<uint32_t>("nEmCalo", 12);
  description.add<uint32_t>("nMu", 2);
  description.addUntracked<bool>("debug", false);
  description.addUntracked<bool>("debug_tk", false);
  description.addUntracked<bool>("debug_calo", false);
  description.addUntracked<bool>("debug_emcalo", false);
  description.addUntracked<bool>("debug_mu", false);
  return description;
}
#endif

l1ct::TDRAltRegionizerEmulator::TDRAltRegionizerEmulator(uint32_t ntk,
                                                         uint32_t ncalo,
                                                         uint32_t nem,
                                                         uint32_t nmu,
                                                         bool debug_tk = false,
                                                         bool debug_calo = false,
                                                         bool debug_emcalo = false,
                                                         bool debug_mu = false
                                                        )
  : RegionizerEmulator(),
    ntk_(ntk),
    ncalo_(ncalo),
    nem_(nem),
    nmu_(nmu),
    init_(false),
    tkRegionizers_(ntk, debug_tk),
    hadCaloRegionizers_(ncalo, debug_calo),
    emCaloRegionizers_(nem, debug_emcalo),
    muRegionizers_(nmu, debug_mu) {}

l1ct::TDRAltRegionizerEmulator::~TDRAltRegionizerEmulator() {}

void l1ct::TDRAltRegionizerEmulator::initSectorsAndRegions(const RegionizerDecodedInputs& in,
                                                           const std::vector<PFInputRegion>& out) {
  if (debug_) {
    dbgCout() << "doing init, out_size = " << out.size() << std::endl;
  }
  assert(!init_);
  init_ = true;


  if (debug_) {
    dbgCout() << "in.track.size() = " << in.track.size() << std::endl;
    dbgCout() << "in.hadcalo.size() = " << in.hadcalo.size() << std::endl;
    dbgCout() << "in.emcalo.size() = " << in.emcalo.size() << std::endl;
  }

  tkRegionizers_.initSectors(in.track);
  tkRegionizers_.initRegions(out);
  hadCaloRegionizers_.initSectors(in.hadcalo);
  hadCaloRegionizers_.initRegions(out);
  emCaloRegionizers_.initSectors(in.emcalo);
  emCaloRegionizers_.initRegions(out);
}

void l1ct::TDRAltRegionizerEmulator::run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) {
  if (debug_) {
    dbgCout() << "TDRAltRegionizerEmulator::run called, out.size =  " << out.size() << std::endl;
  }

  if (!init_) {
    initSectorsAndRegions(in, out);
  }

  //add objects from link
  tkRegionizers_.fillBuffers(in.track);
  tkRegionizers_.run();

  emCaloRegionizers_.fillBuffers(in.emcalo);
  emCaloRegionizers_.run();

  hadCaloRegionizers_.fillBuffers(in.hadcalo);
  hadCaloRegionizers_.run();

  muRegionizers_.fillBuffers(in.muon);
  muRegionizers_.run();


  auto trackSRs = tkRegionizers_.smallRegions();
  auto emCaloSRs = emCaloRegionizers_.smallRegions();
  auto hadCaloSRs = hadCaloRegionizers_.smallRegions();
  auto muSRs = muRegionizers_.smallRegions();

  for (size_t sr = 0; sr < trackSRs.size(); sr++) {
    out[sr].track = trackSRs[sr];
    out[sr].emcalo = emCaloSRs[sr];
    out[sr].hadcalo = hadCaloSRs[sr];
    out[sr].muon = muSRs[sr];
  }

  tkRegionizers_.clearSmallRegions();
  emCaloRegionizers_.clearSmallRegions();
  hadCaloRegionizers_.clearSmallRegions();
  muRegionizers_.clearSmallRegions();
}
