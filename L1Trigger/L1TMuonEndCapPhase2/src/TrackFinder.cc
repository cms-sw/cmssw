#include <iostream>
#include <sstream>
#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/SectorProcessor.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPCollectors.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/CSCTPCollector.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/RPCTPCollector.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/GEMTPCollector.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/ME0TPCollector.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/GE0TPCollector.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/TrackFinder.h"

using namespace emtf::phase2;

TrackFinder::TrackFinder(const edm::ParameterSet& i_config, edm::ConsumesCollector&& i_consumes_collector)
    : context_(i_config, i_consumes_collector), tp_collectors_(), sector_processors_() {
  // ===========================================================================
  // Emulation Setup
  // ===========================================================================

  // Register Trigger Primitives
  if (this->context_.config_.csc_en_) {
    tp_collectors_.push_back(std::make_unique<CSCTPCollector>(context_, i_consumes_collector));
  }

  if (this->context_.config_.rpc_en_) {
    tp_collectors_.push_back(std::make_unique<RPCTPCollector>(context_, i_consumes_collector));
  }

  if (this->context_.config_.gem_en_) {
    tp_collectors_.push_back(std::make_unique<GEMTPCollector>(context_, i_consumes_collector));
  }

  if (this->context_.config_.me0_en_) {
    tp_collectors_.push_back(std::make_unique<ME0TPCollector>(context_, i_consumes_collector));
  }

  if (this->context_.config_.ge0_en_) {
    tp_collectors_.push_back(std::make_unique<GE0TPCollector>(context_, i_consumes_collector));
  }

  // Register Sector Processor
  for (int endcap = kMinEndcap; endcap <= kMaxEndcap; ++endcap) {
    for (int sector = kMinTrigSector; sector <= kMaxTrigSector; ++sector) {
      sector_processors_.push_back(std::make_unique<SectorProcessor>(context_, endcap, sector));
    }
  }
}

TrackFinder::~TrackFinder() {
  // Do Nothing
}

void TrackFinder::process(
    // Input
    const edm::Event& i_event,
    const edm::EventSetup& i_event_setup,
    // Output
    EMTFHitCollection& out_hits,
    EMTFTrackCollection& out_tracks,
    EMTFInputCollection& out_inputs) {
  // ===========================================================================
  // Clear output collections
  // ===========================================================================

  out_hits.clear();
  out_tracks.clear();
  out_inputs.clear();

  // ===========================================================================
  // Load the event configuration
  // ===========================================================================

  context_.update(i_event, i_event_setup);

  // ===========================================================================
  // Collect trigger primitives
  // ===========================================================================

  // Build BX Sequence
  std::vector<int> bx_sequence;

  {
    auto min_bx = this->context_.config_.min_bx_;
    auto delay_bx = this->context_.config_.bx_window_ - 1;
    auto max_bx = this->context_.config_.max_bx_ + delay_bx;

    for (int bx = min_bx; bx <= max_bx; ++bx) {
      bx_sequence.push_back(bx);
    }
  }

  // Collect TP per BX
  BXTPCMap bx_tpc_map;

  for (auto& tp_collector : tp_collectors_) {
    tp_collector->collect(i_event, bx_tpc_map);
  }

  // Debug Info
  if (this->context_.config_.verbosity_ > 4) {
    int n_tp = 0;

    // Loop BX
    for (const auto& bx : bx_sequence) {
      // Get trigger primitives for this BX
      auto bx_tpc_map_it = bx_tpc_map.find(bx);
      auto bx_tpc_map_end = bx_tpc_map.end();

      // Short-Circuit: Empty trigger primitive collection
      if (bx_tpc_map_it == bx_tpc_map_end) {
        continue;
      }

      // Reference TPC
      auto& bx_tpc = bx_tpc_map_it->second;

      // Short-Circuit: Empty trigger primitive collection
      if (bx_tpc.empty()) {
        continue;
      }

      // Print trigger primitives
      edm::LogInfo("L1TEMTFpp") << "==========================================================================="
                                << std::endl;
      edm::LogInfo("L1TEMTFpp") << "Begin TPC BX " << bx << " Dump" << std::endl;
      edm::LogInfo("L1TEMTFpp") << "---------------------------------------------------------------------------"
                                << std::endl;

      n_tp += bx_tpc.size();

      for (const auto& tp_entry : bx_tpc) {
        tp_entry.tp_.print(std::cout);

        edm::LogInfo("L1TEMTFpp") << "---------------------------------------------------------------------------"
                                  << std::endl;
      }

      edm::LogInfo("L1TEMTFpp") << "End TPC BX " << bx << " Dump" << std::endl;
      edm::LogInfo("L1TEMTFpp") << "==========================================================================="
                                << std::endl;
    }

    // Print TPrimitives Summary
    if (n_tp > 0) {
      edm::LogInfo("L1TEMTFpp") << "Num of TriggerPrimitive: " << n_tp << std::endl;
      edm::LogInfo("L1TEMTFpp") << "==========================================================================="
                                << std::endl;
    }
  }

  // ===========================================================================
  // Run sector processors
  // ===========================================================================

  // Before event
  for (auto& sector_processor : sector_processors_) {
    sector_processor->configure_event(i_event);
  }

  // Orderly loop BX
  for (const auto& bx : bx_sequence) {
    // Get trigger primitives for this BX
    auto bx_tpc_map_it = bx_tpc_map.find(bx);
    auto bx_tpc_map_end = bx_tpc_map.end();

    TPCollection* bx_tpc_ptr = nullptr;

    if (bx_tpc_map_it != bx_tpc_map_end) {
      bx_tpc_ptr = &(bx_tpc_map_it->second);
    }

    // Loop over all sector processors
    for (auto& sector_processor : sector_processors_) {
      // Before BX
      sector_processor->configure_bx(bx);

      // Select trigger primitives in BX
      if (bx_tpc_ptr != nullptr) {
        for (const auto& tp_entry : *bx_tpc_ptr) {
          const auto& tp = tp_entry.tp_;
          const auto& tp_info = tp_entry.info_;

          sector_processor->select(tp, tp_info);
        }
      }

      // Process trigger primitives
      sector_processor->process(out_hits, out_tracks, out_inputs);
    }

    // Free memory: Removes BX TPCollections after all Sector Processors have selected their TPrimitives
    if (bx_tpc_ptr != nullptr) {
      bx_tpc_map.erase(bx_tpc_map_it);
    }
  }

  // Free memory: Drops any BX TPCollections outside of the [min bx, max bx] range
  bx_tpc_map.clear();
}

void TrackFinder::on_job_begin() {
  // Do Nothing
}

void TrackFinder::on_job_end() {
  // Do Nothing
}
