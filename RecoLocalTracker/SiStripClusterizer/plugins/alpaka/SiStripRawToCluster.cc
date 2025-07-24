#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClusterDevice.h"
#include "DataFormats/SiStripDigiSoA/interface/alpaka/SiStripDigiDevice.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDeviceObject.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsRecord.h"
#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"

#include "SiStripRawToClusterAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;

  class SiStripRawToCluster : public stream::SynchronizingEDProducer<> {
  public:
    SiStripRawToCluster(edm::ParameterSet const& iConfig);
    ~SiStripRawToCluster() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    std::unique_ptr<PortableFEDMover> fillFedIdFedChBuffer(Queue& queue,
                                                           const SiStripClusterizerConditions& stripCond,
                                                           const FEDRawDataCollection& rawColl);

    // RAW unpacking and clustering algorithm
    SiStripRawToClusterAlgo algo_;

    edm::EDGetTokenT<FEDRawDataCollection> fedRawGetToken_;
    edm::ESGetToken<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd> stripCondGetToken_;
    device::ESGetToken<SiStripClusterizerConditionsDetToFedsDeviceObject, SiStripClusterizerConditionsDetToFedsRecord>
        stripCablCondGetToken_;
    device::ESGetToken<SiStripClusterizerConditionsGainNoiseCalsDeviceObject,
                       SiStripClusterizerConditionsGainNoiseCalsRecord>
        stripDataCondGetToken_;
    device::EDPutToken<SiStripClusterDevice> stripClustPutToken_;
    device::EDPutToken<SiStripDigiDevice> stripDigiPutToken_;

    // Setup
    bool doAPVEmulatorCheck_;

    // Helper functions to fill valid, condition-passing raw/buffers
    std::unique_ptr<FEDBuffer> fillBuffer(int fedId, const FEDRawData& rawData);
  };

  SiStripRawToCluster::SiStripRawToCluster(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        algo_(iConfig.getParameter<edm::ParameterSet>("Clusterizer")),
        fedRawGetToken_(consumes(iConfig.getParameter<edm::InputTag>("ProductLabel"))),
        stripCondGetToken_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("ConditionsLabel")})),
        stripCablCondGetToken_(
            esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("CablingConditionsLabel")})),
        stripDataCondGetToken_(esConsumes()),
        stripClustPutToken_(produces()),
        stripDigiPutToken_(produces()),
        doAPVEmulatorCheck_(iConfig.getParameter<bool>("DoAPVEmulatorCheck")) {}

  void SiStripRawToCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    // Add some custom parameter description to the automatically-created ones by the addWithDefaultLabel methos
    edm::ParameterSetDescription desc;
    desc.add("ProductLabel", edm::InputTag("rawDataCollector"));
    desc.add<std::string>("ConditionsLabel", "");
    desc.add<std::string>("CablingConditionsLabel", "");

    // Setup parameters (from SiStripClusterizerFromRaw)
    desc.add<bool>("DoAPVEmulatorCheck", false);

    // Inherit all the parameters from the clusterizers (all var.s from the Clusterizer PSet)
    // 1:Algorithm, ConditionsLabel, ChannelThreshold, SeedThreshold, ClusterThreshold,
    // MaxSequentialHoles, MaxSequentialBad, MaxAdjacentBad, MaxClusterSize, RemoveApvShots,
    // setDetId, 12:clusterChargeCut
    edm::ParameterSetDescription clusterizer;
    StripClusterizerAlgorithmFactory::fillDescriptions(clusterizer);
    // The parameter MaxSeedStrips determines the size for pre-allocation of the clusters collection SoA
    clusterizer.add<uint32_t>("MaxSeedStrips", 200000u);
    desc.add("Clusterizer", clusterizer);

    descriptions.addWithDefaultLabel(desc);
  }

  void SiStripRawToCluster::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    // If this is the first time the module is called / the record signalled a change in conditions,
    // then load/update the conditions + cabling map

    // Host conditions (for buffer preconstruct + fedCh mapping preparation)
    const auto& stripCond = iSetup.getData(stripCondGetToken_);

    // Device conditions (cabling map)
    const auto& stripCablCond = iSetup.getData(stripCablCondGetToken_);
    LogDebug("SiStripRawToCluster") << "#sizeB,stripCablCond_," << alpaka::getExtentProduct(stripCablCond.buffer());

    // Get FED raw data collection
    const auto& rawCollection = iEvent.get(fedRawGetToken_);

    // Prepare the LUT for FEDCh mapping in the raw[] on the device
    std::unique_ptr<PortableFEDMover> fedChMover = fillFedIdFedChBuffer(iEvent.queue(), stripCond, rawCollection);

    // Move PortableFEDMover class to algo
    algo_.prepareUnpackCluster(iEvent.queue(), stripCablCond.const_data(), std::move(fedChMover));
  }

  void SiStripRawToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    // Device conditions (strip noise)
    const auto& stripDataCond = iSetup.getData(stripDataCondGetToken_);
    LogDebug("SiStripRawToCluster") << "#sizeB,stripDataCond_," << alpaka::getExtentProduct(stripDataCond.buffer());

    // Unpack the raw FED data into strip digi
    auto nStrips = algo_.unpackStrips(iEvent.queue(), stripDataCond.const_data());
    if (nStrips == 0) {
      // No strips to unpack, empty cluster collection
      iEvent.emplace(stripClustPutToken_, 0, iEvent.queue());
      iEvent.emplace(stripDigiPutToken_, 0, iEvent.queue());
    } else {
      // Run the clusterization algorithm (ThreeThresholdAlgorithm)
      auto cluster_d = algo_.makeClusters(iEvent.queue(), stripDataCond.const_data());

      // Get the clusters amplitudes
      auto clusterAmpls_d = algo_.releaseDigiAmplitudes();

      iEvent.put(stripClustPutToken_, std::move(cluster_d));
      iEvent.put(stripDigiPutToken_, std::move(clusterAmpls_d));
    }
  }

  std::unique_ptr<PortableFEDMover> SiStripRawToCluster::fillFedIdFedChBuffer(
      Queue& queue, const SiStripClusterizerConditions& stripCond, const FEDRawDataCollection& rawColl) {
    // Containers for the condition-passing raw data
    const auto fedID_maxIdx = sistrip::FED_ID_MAX + 1;
    std::vector<const FEDRawData*> raw(fedID_maxIdx, nullptr);
    std::vector<std::unique_ptr<FEDBuffer>> buffers(fedID_maxIdx);

    std::vector<FEDChMetadata> fedChOfs_wrt_rawFedId_;
    uint32_t rawBuffFlattenSize = 0;

    // loop over good det in cabling
    for (auto idet : stripCond.allDetIds()) {
      // it populates raw, buffers with only connected fed
      auto const& det = stripCond.findDetId(idet);
      if (!det.valid()) {
        continue;  // idet loop
      }

      // Loop over apv-pairs of det
      // int connIdx = 0;
      for (auto const conn : stripCond.currentConnection(det)) {
        // connIdx++;
        if (!conn) {
          continue;  // conn loop
        }

        const uint16_t fedId = conn->fedId();

        // If fed id is null or connection is invalid continue
        if (!fedId || !conn->isConnected()) {
          continue;  // conn loop
        }

        if (!raw[fedId]) {
          // pointer to the raw data in the collection
          raw[fedId] = &rawColl.FEDData(fedId);
          rawBuffFlattenSize += raw[fedId]->size();
        }

        // If Fed hasnt already been initialised, extract data and initialise
        sistrip::FEDBuffer* buffer = buffers[fedId].get();
        if (!buffer) {
          buffers[fedId] = fillBuffer(fedId, *raw[fedId]);
          if (!buffers[fedId]) {
            continue;
          } else {
            buffer = buffers[fedId].get();
          }
        }
        assert(buffer);

        // Set legacy unpacking
        // buffers[fedidx]->setLegacyMode(legacyUnpacker_);

        // Check the readout mode of the buffer
        const FEDReadoutMode buffROMode = buffer->readoutMode();
        // Make sure EACH buffer has a readout mode supported by the module
        if (!(buffROMode >= READOUT_MODE_ZERO_SUPPRESSED_LITE10 &&
              buffROMode <= READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE &&
              buffROMode != READOUT_MODE_PROC_RAW)) {
          if (edm::isDebugEnabled()) {
            std::ostringstream ss;
            ss << "Unsupported buffer ROmode=" << buffROMode << " fedID= " << fedId;
            edm::LogWarning("fillFedIdFedChBuffer") << ss.str();
          }
          continue;  // conn loop
        }

        // check channel
        const uint8_t fedCh = conn->fedCh();
        if (!buffer->channelGood(fedCh, doAPVEmulatorCheck_)) {
          if (edm::isDebugEnabled()) {
            std::ostringstream ss;
            ss << "Problem unpacking channel " << (int)fedCh << " on FED " << fedId;
            edm::LogWarning("fillFedIdFedChBuffer") << ss.str();
          }
          continue;  // conn loop
        }

        // Good fedId and fedCh in the raw data
        const FEDChannel& channel = buffer->channel(fedCh);
        const uint32_t fedChOfs = channel.offset();

        auto diff = channel.data() - raw[fedId]->data();
        if (diff < 0 || diff > std::numeric_limits<uint32_t>::max()) {
          if (edm::isDebugEnabled()) {
            std::ostringstream ss;
            ss << "Large diff " << diff << " for fedId " << fedId << " fedCh " << fedCh;
            edm::LogWarning("fillFedIdFedChBuffer") << ss.str();
          }
          continue;
        }
        const uint32_t fedChOfs_wrt_rawFedId = static_cast<const uint32_t>(diff);

        fedChOfs_wrt_rawFedId_.push_back(
            FEDChMetadata(idet, fedId, fedCh, fedChOfs, fedChOfs_wrt_rawFedId, buffROMode));
      }
    }

    // Do I have any channel to unpack ( FEDRaw data empty || mapping empty) ?
    if (fedChOfs_wrt_rawFedId_.empty()) {
      return nullptr;
    }

    // Create container for the buffer and mapping
    auto fedMover = std::make_unique<PortableFEDMover>(queue, rawBuffFlattenSize, fedChOfs_wrt_rawFedId_.size());

    // Fill buffer
    fedMover->fillBuffer(raw);

    // Fill the mapping
    fedMover->fillMapping(fedChOfs_wrt_rawFedId_);

    return fedMover;
  }

  std::unique_ptr<sistrip::FEDBuffer> SiStripRawToCluster::fillBuffer(int fedId, const FEDRawData& rawData) {
    std::unique_ptr<sistrip::FEDBuffer> buffer;

    // Check on FEDRawData pointer
    const auto st_buffer = sistrip::preconstructCheckFEDBuffer(rawData);
    if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_buffer) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
            << "[ClustersFromRawProducer::" << __func__ << "]" << st_buffer << " for FED ID " << fedId;
      }
      return buffer;
    }
    buffer = std::make_unique<sistrip::FEDBuffer>(rawData);
    const auto st_chan = buffer->findChannels();
    if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_chan) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
            << "Exception caught when creating FEDBuffer object for FED " << fedId << ": " << st_chan;
      }
      buffer.reset();
      return buffer;
    }
    if UNLIKELY (!buffer->doChecks(false)) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
            << "Exception caught when creating FEDBuffer object for FED " << fedId << ": FED Buffer check fails";
      }
      buffer.reset();
      return buffer;
    }

    return buffer;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(sistrip::SiStripRawToCluster);
