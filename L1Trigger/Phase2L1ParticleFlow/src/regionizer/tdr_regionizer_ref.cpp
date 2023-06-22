#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/tdr_regionizer_elements_ref.icc"

#include <iostream>

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"

l1ct::TDRRegionizerEmulator::TDRRegionizerEmulator(const edm::ParameterSet& iConfig)
    : TDRRegionizerEmulator(iConfig.getParameter<uint32_t>("nTrack"),
                            iConfig.getParameter<uint32_t>("nCalo"),
                            iConfig.getParameter<uint32_t>("nEmCalo"),
                            iConfig.getParameter<uint32_t>("nMu"),
                            iConfig.getParameter<uint32_t>("nClocks"),
                            iConfig.getParameter<std::vector<int32_t>>("bigRegionEdges"),
                            iConfig.getParameter<bool>("doSort")) {
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
}
#endif

l1ct::TDRRegionizerEmulator::TDRRegionizerEmulator(uint32_t ntk,
                                                   uint32_t ncalo,
                                                   uint32_t nem,
                                                   uint32_t nmu,
                                                   uint32_t nclocks,
                                                   std::vector<int32_t> bigRegionEdges,
                                                   bool dosort)
    : RegionizerEmulator(),
      ntk_(ntk),
      ncalo_(ncalo),
      nem_(nem),
      nmu_(nmu),
      nclocks_(nclocks),
      bigRegionEdges_(bigRegionEdges),
      dosort_(dosort),
      netaInBR_(6),
      nphiInBR_(3),
      init_(false) {
  nBigRegions_ = bigRegionEdges_.size() - 1;
}

l1ct::TDRRegionizerEmulator::~TDRRegionizerEmulator() {}

void l1ct::TDRRegionizerEmulator::initSectorsAndRegions(const RegionizerDecodedInputs& in,
                                                        const std::vector<PFInputRegion>& out) {
  if (debug_) {
    dbgCout() << "doing init, out_size = " << out.size() << std::endl;
  }
  assert(!init_);
  init_ = true;

  for (unsigned int i = 0; i < nBigRegions_; i++) {
    tkRegionizers_.emplace_back(
        netaInBR_, nphiInBR_, ntk_, bigRegionEdges_[i], bigRegionEdges_[i + 1], nclocks_, 1, false);
    // duplicate input fibers to increase to increasee the throughput, since lots of data comes in per fiber
    hadCaloRegionizers_.emplace_back(
        netaInBR_, nphiInBR_, ncalo_, bigRegionEdges_[i], bigRegionEdges_[i + 1], nclocks_, 2, false);
    emCaloRegionizers_.emplace_back(
        netaInBR_, nphiInBR_, nem_, bigRegionEdges_[i], bigRegionEdges_[i + 1], nclocks_, 1, false);
    muRegionizers_.emplace_back(
        netaInBR_, nphiInBR_, nmu_, bigRegionEdges_[i], bigRegionEdges_[i + 1], nclocks_, 1, false);
  }

  dbgCout() << "in.track.size() = " << in.track.size() << std::endl;
  dbgCout() << "in.hadcalo.size() = " << in.hadcalo.size() << std::endl;
  dbgCout() << "in.emcalo.size() = " << in.emcalo.size() << std::endl;

  if (ntk_) {
    for (unsigned int i = 0; i < nBigRegions_; i++) {
      tkRegionizers_[i].initSectors(in.track);
      tkRegionizers_[i].initRegions(out);
    }
  }
  if (ncalo_) {
    for (unsigned int i = 0; i < nBigRegions_; i++) {
      hadCaloRegionizers_[i].initSectors(in.hadcalo);
      hadCaloRegionizers_[i].initRegions(out);
    }
  }
  if (nem_) {
    for (unsigned int i = 0; i < nBigRegions_; i++) {
      emCaloRegionizers_[i].initSectors(in.emcalo);
      emCaloRegionizers_[i].initRegions(out);
    }
  }
  if (nmu_) {
    for (unsigned int i = 0; i < nBigRegions_; i++) {
      muRegionizers_[i].initSectors(in.muon);
      muRegionizers_[i].initRegions(out);
    }
  }
}

void l1ct::TDRRegionizerEmulator::run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) {
  if (debug_) {
    dbgCout() << "TDRRegionizerEmulator::run called, out.size =  " << out.size() << std::endl;
  }

  if (!init_) {
    initSectorsAndRegions(in, out);
  }

  for (unsigned int ie = 0; ie < nBigRegions_; ie++) {
    //add objects from link
    tkRegionizers_[ie].reset();
    tkRegionizers_[ie].fillBuffers(in.track);
    tkRegionizers_[ie].run();

    emCaloRegionizers_[ie].reset();
    emCaloRegionizers_[ie].fillBuffers(in.emcalo);
    emCaloRegionizers_[ie].run();

    hadCaloRegionizers_[ie].reset();
    hadCaloRegionizers_[ie].fillBuffers(in.hadcalo);
    hadCaloRegionizers_[ie].run();

    muRegionizers_[ie].reset();
    muRegionizers_[ie].fillBuffers(in.muon);
    muRegionizers_[ie].run();
  }

  for (unsigned int ie = 0; ie < nBigRegions_; ie++) {
    auto regionTrackMap = tkRegionizers_[ie].fillRegions(dosort_);
    for (auto& pr : regionTrackMap) {
      out[pr.first].track = pr.second;
    }
    auto regionEmCaloMap = emCaloRegionizers_[ie].fillRegions(dosort_);
    for (auto& pr : regionEmCaloMap) {
      out[pr.first].emcalo = pr.second;
    }
    auto regionHadCaloMap = hadCaloRegionizers_[ie].fillRegions(dosort_);
    for (auto& pr : regionHadCaloMap) {
      out[pr.first].hadcalo = pr.second;
    }
    auto regionMuMap = muRegionizers_[ie].fillRegions(dosort_);
    for (auto& pr : regionMuMap) {
      out[pr.first].muon = pr.second;
    }
  }
}
