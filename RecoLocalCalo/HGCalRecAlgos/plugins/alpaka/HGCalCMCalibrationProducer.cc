// HGCal CM calibration using a PyTorch DNN via alpaka.
//
// Per-event workflow:
//   1. Consume host digis and copy to device.
//   2. Compute per-module ntoa/ntot on the host (cheap loop over digis).
//   3. Fill the ML input SoA on the device (one slot per global channel).
//   4. Run DNN inference:  inputs  → [cmsum tensor, features tensor]
//                          outputs → [correction tensor]
//   5. Apply per-channel additive correction to digi ADC in-place.
//
// Open-question assumptions encoded here:
//   - DNN model signature: forward(cmsum:[N,12], features:[N,6]) -> correction:[N]
//   - Correction is an additive float offset to raw digi ADC (uint16_t, clamped).
//   - cellfrac sourced from HGCalMappingCellParamSoA::trace().

#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"

#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalCalibrationParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/alpaka/HGCalCMCalibrationAlgorithms.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class HGCalCMCalibrationProducer : public stream::EDProducer<> {
  public:
    explicit HGCalCMCalibrationProducer(const edm::ParameterSet& iConfig);
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(device::Event&, device::EventSetup const&) override;

    // --- tokens ---
    const edm::EDGetTokenT<hgcaldigi::HGCalDigiHost> digisToken_;
    const edm::EDPutTokenT<hgcaldigi::HGCalDigiHost> correctedDigisToken_;
    const edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIndexerToken_;
    const device::ESGetToken<hgcal::HGCalDenseIndexInfoDevice, HGCalDenseIndexInfoRcd> indexingToken_;
    const device::ESGetToken<hgcal::HGCalMappingCellParamDevice, HGCalElectronicsMappingRcd> cellmapToken_;
    const edm::ESGetToken<hgcalrechit::HGCalCalibParamHost, HGCalModuleConfigurationRcd> calibToken_;

    // --- per-event device buffers for flat module arrays (rebuilt if ES changes) ---
    // These are re-uploaded every event for simplicity; a watcher can be added later.
    // Alternative: use an ESProducer to cache these on device — kept simple here.

    // --- algorithm and DNN model ---
    const HGCalCMCalibrationAlgorithms algo_;
    torch::AlpakaModel model_;

    // --- cell area scale factors from cellareas.json, uploaded to device each event ---
    std::vector<float> h_sfLD_;  // ML_F SF, indexed by chIdx (222 entries, 6 eRx × 37)
    std::vector<float> h_sfHD_;  // MH_F SF, indexed by chIdx (444 entries, 12 eRx × 37)

    // --- debug print filter (0 = disabled for event; UINT32_MAX = all modules) ---
    // Module can be specified either as a dense index (debugModule) or by its
    // hardware address (debugFedId + debugCaptureBlock + debugEcond).  If all
    // three hardware fields are set (i.e. != UINT32_MAX), they take precedence.
    uint64_t debug_event_;
    uint32_t debug_module_;
    uint32_t debug_max_ch_;
    uint32_t debug_fed_id_;
    uint32_t debug_capblock_;
    uint32_t debug_econd_;
  };

  // ---------------------------------------------------------------------------
  HGCalCMCalibrationProducer::HGCalCMCalibrationProducer(const edm::ParameterSet& iConfig)
      : EDProducer<>(iConfig),
        digisToken_{consumes<hgcaldigi::HGCalDigiHost>(iConfig.getParameter<edm::InputTag>("digis"))},
        correctedDigisToken_{produces().produces<hgcaldigi::HGCalDigiHost>()},
        moduleIndexerToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("moduleIndexerSource"))},
        indexingToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("indexingSource"))},
        cellmapToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("cellmapSource"))},
        calibToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("calibSource"))},
        algo_{iConfig.getParameter<int>("n_threads")},
        model_{iConfig.getParameter<edm::FileInPath>("model").fullPath()},
        debug_event_{uint64_t(iConfig.getParameter<unsigned int>("debugEvent"))},
        debug_module_{iConfig.getParameter<unsigned int>("debugModule")},
        debug_max_ch_{iConfig.getParameter<unsigned int>("debugMaxChannels")},
        debug_fed_id_{iConfig.getParameter<unsigned int>("debugFedId")},
        debug_capblock_{iConfig.getParameter<unsigned int>("debugCaptureBlock")},
        debug_econd_{iConfig.getParameter<unsigned int>("debugEcond")} {
    // Load cell area scale factors from cellareas.json.
    // JSON structure: { "ML_F": { "SF": [...222 floats...] }, "MH_F": { "SF": [...444 floats...] } }
    std::string cellAreasPath = iConfig.getParameter<edm::FileInPath>("cellAreas").fullPath();
    std::ifstream cellAreasFile(cellAreasPath);
    if (!cellAreasFile.is_open())
      throw cms::Exception("Configuration") << "Cannot open cellareas file: " << cellAreasPath;
    nlohmann::json cellAreasJson;
    cellAreasFile >> cellAreasJson;
    h_sfLD_ = cellAreasJson.at("ML_F").at("SF").get<std::vector<float>>();
    h_sfHD_ = cellAreasJson.at("MH_F").at("SF").get<std::vector<float>>();
    LogDebug("HGCalCMCalibrationProducer")
        << "Loaded cellareas: ML_F=" << h_sfLD_.size() << " SF entries, "
        << "MH_F=" << h_sfHD_.size() << " SF entries";
  }

  // ---------------------------------------------------------------------------
  void HGCalCMCalibrationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("digis", edm::InputTag("hgcalDigis", "DIGI"))
        ->setComment("HGCal digi host collection");
    desc.add<edm::ESInputTag>("moduleIndexerSource", edm::ESInputTag(""))
        ->setComment("HGCalMappingModuleIndexer for channel/module layout");
    desc.add<edm::ESInputTag>("indexingSource", edm::ESInputTag(""))
        ->setComment("HGCalDenseIndexInfoDevice for per-channel mapping");
    desc.add<edm::ESInputTag>("cellmapSource", edm::ESInputTag(""))
        ->setComment("HGCalMappingCellParamDevice for cell area fraction");
    desc.add<edm::ESInputTag>("calibSource", edm::ESInputTag(""))
        ->setComment("HGCalCalibParamHost for per-channel ADC_ped and CM_ped");
    desc.add<edm::FileInPath>("model")->setComment("Path to TorchScript DNN model (.pth)");
    desc.add<edm::FileInPath>("cellAreas")->setComment("Path to cellareas.json with per-channel SF (ML_F/MH_F)");
    desc.add<int>("n_threads", 256)->setComment("Threads per alpaka block");
    desc.add<unsigned int>("debugEvent", 0u)
        ->setComment("Event number to print DNN inputs for (0 = disabled)");
    desc.add<unsigned int>("debugModule", ~0u)
        ->setComment("Dense module index to filter (UINT32_MAX = all); overridden by debugFedId/CaptureBlock/Econd");
    desc.add<unsigned int>("debugMaxChannels", ~0u)
        ->setComment("Max within-module chIdx to print (UINT32_MAX = all; set to e.g. 5 for first 5 channels)");
    desc.add<unsigned int>("debugFedId", ~0u)
        ->setComment("FED ID of module to debug (must set all three hardware params to take effect)");
    desc.add<unsigned int>("debugCaptureBlock", ~0u)
        ->setComment("Capture block index of module to debug");
    desc.add<unsigned int>("debugEcond", ~0u)
        ->setComment("ECON-D index of module to debug");
    descriptions.addWithDefaultLabel(desc);
  }

  // ---------------------------------------------------------------------------
  void HGCalCMCalibrationProducer::produce(device::Event& iEvent,
                                            device::EventSetup const& iSetup) {
    auto& queue = iEvent.queue();

    // ---- Retrieve conditions ----
    const auto& moduleIndexer = iSetup.getData(moduleIndexerToken_);
    const auto& deviceIndex = iSetup.getData(indexingToken_);
    const auto& deviceCellmap = iSetup.getData(cellmapToken_);
    const auto& hostCalib = iSetup.getData(calibToken_);

    // maxModulesCount() = total physical modules; maxModuleSize() = distinct typecodes (often 1).
    // Module indices (modOffsets_) run 0..maxModulesCount()-1, so size the arrays by count.
    const uint32_t nmodules = moduleIndexer.maxModulesCount();

    // ---- Build flat per-module arrays on host, copy to device ----
    // chDataOffsets[denseModIdx] = first global channel index for that module
    // enabledErx[denseModIdx]   = eRx bitmask for that module (bit e set → eRx e is active)
    std::vector<uint32_t> h_chDataOffsets(nmodules, 0u);
    std::vector<uint32_t> h_enabledErx(nmodules, 0u);

    for (const auto& fed : moduleIndexer.fedReadoutSequences()) {
      for (uint32_t modid = 0; modid < fed.chDataOffsets_.size(); ++modid) {
        uint32_t denseModIdx = moduleIndexer.getIndexForModule(fed.id, modid);
        if (denseModIdx >= nmodules)
          continue;
        h_chDataOffsets[denseModIdx] = fed.chDataOffsets_[modid];
        h_enabledErx[denseModIdx] = fed.enabledErx_[modid];
      }
    }

    auto d_chDataOffsets = make_device_buffer<uint32_t[]>(queue, nmodules);
    auto d_enabledErx = make_device_buffer<uint32_t[]>(queue, nmodules);
    alpaka::memcpy(queue, d_chDataOffsets, make_host_view(h_chDataOffsets.data(), nmodules));
    alpaka::memcpy(queue, d_enabledErx, make_host_view(h_enabledErx.data(), nmodules));

    // Upload cell area SF tables to device.
    auto d_sfLD = make_device_buffer<float[]>(queue, h_sfLD_.size());
    auto d_sfHD = make_device_buffer<float[]>(queue, h_sfHD_.size());
    alpaka::memcpy(queue, d_sfLD, make_host_view(h_sfLD_.data(), h_sfLD_.size()));
    alpaka::memcpy(queue, d_sfHD, make_host_view(h_sfHD_.data(), h_sfHD_.size()));

    // Upload per-channel pedestals (ADC_ped for unconn channels, CM_ped for CM sums).
    const uint32_t ncalib = uint32_t(hostCalib.view().metadata().size());
    std::vector<float> h_adcPed(ncalib), h_cmPed(ncalib);
    for (uint32_t i = 0; i < ncalib; ++i) {
      h_adcPed[i] = hostCalib.view()[i].ADC_ped();
      h_cmPed[i]  = hostCalib.view()[i].CM_ped();
    }
    auto d_adcPed = make_device_buffer<float[]>(queue, ncalib);
    auto d_cmPed  = make_device_buffer<float[]>(queue, ncalib);
    alpaka::memcpy(queue, d_adcPed, make_host_view(h_adcPed.data(), ncalib));
    alpaka::memcpy(queue, d_cmPed,  make_host_view(h_cmPed.data(),  ncalib));

    // ---- Retrieve and copy digis to device ----
    const auto& hostDigis = iEvent.get(digisToken_);
    const uint32_t ndigis = hostDigis.view().metadata().size();
    hgcaldigi::HGCalDigiDevice deviceDigis(queue, ndigis);
    alpaka::memcpy(queue, deviceDigis.buffer(), hostDigis.const_buffer());

    // ---- Compute per-module ntoa / ntot on the host ----
    // Match the reference definition in DigiAnalysisUtils.py:
    //   ntot: flags!=NotAvailable && tctp==3 && tot>0   (genuine TOT-mode hits only)
    //   ntoa: flags!=NotAvailable && toa>0
    // Without the tctp==3 gate, ADC-mode channels with stale tot fields inflate ntot.
    //
    // Counts are accumulated per module (indexed by denseModIdx) rather than over the whole
    // event: each module's channels are contiguous in the digi SoA, starting at
    // chDataOffsets[m] and spanning popcount(enabledErx[m])*37 channels (nErx eRx × 37 ch).
    // The kernel then broadcasts d_ntoa[denseModIdx]/d_ntot[denseModIdx] to every slot of
    // that module.  This reuses the host arrays already built above; no digi→module lookup
    // or device atomics are needed.
    std::vector<int> h_ntoa(nmodules, 0), h_ntot(nmodules, 0);
    for (uint32_t m = 0; m < nmodules; ++m) {
      const uint32_t base = h_chDataOffsets[m];
      const uint32_t nch = uint32_t(__builtin_popcount(h_enabledErx[m])) * 37u;
      for (uint32_t i = base; i < base + nch && i < ndigis; ++i) {
        if (hostDigis.view()[i].flags() == ::hgcal::DIGI_FLAG::NotAvailable)
          continue;
        h_ntoa[m] += (hostDigis.view()[i].toa() > 0) ? 1 : 0;
        h_ntot[m] += (hostDigis.view()[i].tctp() == 3 && hostDigis.view()[i].tot() > 0) ? 1 : 0;
      }
    }
    auto d_ntoa = make_device_buffer<int[]>(queue, nmodules);
    auto d_ntot = make_device_buffer<int[]>(queue, nmodules);
    alpaka::memcpy(queue, d_ntoa, make_host_view(h_ntoa.data(), nmodules));
    alpaka::memcpy(queue, d_ntot, make_host_view(h_ntot.data(), nmodules));
    LogDebug("HGCalCMCalibrationProducer") << "computed per-module ntoa/ntot for "
                                           << nmodules << " modules, ndigis=" << ndigis;

    // Resolve hardware module address → dense index if all three fields are set.
    uint32_t eff_debug_module = debug_module_;
    if (debug_fed_id_ != ~0u && debug_capblock_ != ~0u && debug_econd_ != ~0u) {
      eff_debug_module = moduleIndexer.getIndexForModule(
          debug_fed_id_, uint16_t(debug_capblock_), uint16_t(debug_econd_));
      LogDebug("HGCalCMCalibrationProducer")
          << "debug: FED=" << debug_fed_id_ << " capblock=" << debug_capblock_
          << " econd=" << debug_econd_ << " → denseModIdx=" << eff_debug_module;
    }

    // ---- Allocate ML input SoA on device and fill (one slot per digi) ----
    HGCalSoACMMLDeviceCollection deviceMLSoA(queue, ndigis);
    algo_.fillCMInputs(queue,
                       ndigis,
                       d_ntoa.data(),
                       d_ntot.data(),
                       deviceDigis,
                       deviceIndex,
                       deviceCellmap,
                       d_chDataOffsets.data(),
                       d_enabledErx.data(),
                       d_sfLD.data(),
                       d_sfHD.data(),
                       d_adcPed.data(),
                       d_cmPed.data(),
                       uint64_t(iEvent.id().event()),
                       debug_event_,
                       eff_debug_module,
                       debug_max_ch_,
                       deviceMLSoA);

    // ---- DNN inference ----
    // All 21 input columns are adjacent float SOA_COLUMNs → contiguous [ndigis, 21] tensor.
    // Model: forward(features:[N,21]) -> correction:[N,1]  (subtractive: corrected = raw - pred)
    HGCalCMCorrectionDeviceCollection deviceCorrections(queue, ndigis);

    auto in_records = deviceMLSoA.const_view().records();
    auto out_records = deviceCorrections.view().records();

    const int batchSize = static_cast<int>(ndigis);

    cms::torch::alpakatools::TensorCollection<Queue> inputs{batchSize};
    inputs.template add<hgcalcmml::HGCalCMMLSoA>("features",
        in_records.cm0(),     in_records.cm1(),  in_records.cm2(),  in_records.cm3(),
        in_records.cm4(),     in_records.cm5(),  in_records.cm6(),  in_records.cm7(),
        in_records.cm8(),     in_records.cm9(),  in_records.cm10(), in_records.cm11(),
        in_records.msubchidx(),
        in_records.msuberxidx(),
        in_records.cellfrac(),
        in_records.unconn0(), in_records.unconn1(),
        in_records.unconn2(), in_records.unconn3(),
        in_records.ntoa(),
        in_records.ntot());

    cms::torch::alpakatools::TensorCollection<Queue> outputs{batchSize};
    outputs.template add<hgcalcmml::HGCalCMCorrectionSoA>("correction", out_records.correction());

    model_.forward(queue, inputs, outputs);

    // ---- Subtract DNN prediction from digi ADC in-place ----
    // deviceMLSoA carries the per-digi cellfrac so unconnected channels (SF==0) are skipped.
    algo_.applyCMCorrections(queue, ndigis, deviceCorrections, deviceMLSoA, deviceDigis);

    // ---- Copy corrected digis back to host and emplace into event ----
    // The host collection is allocated and the async memcpy is enqueued here.
    // The framework drains the alpaka queue after produce() returns, so the data
    // is valid by the time any downstream consumer accesses it.
    hgcaldigi::HGCalDigiHost correctedHostDigis(queue, ndigis);
    alpaka::memcpy(queue, correctedHostDigis.buffer(), deviceDigis.const_buffer());
    iEvent.emplace(correctedDigisToken_, std::move(correctedHostDigis));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalCMCalibrationProducer);
