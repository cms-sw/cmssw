#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
// #include "HeterogeneousCore/AlpakaInterface/interface/moveToDeviceAsync.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"
#include "DataFormats/SiStripDigiSoA/interface/alpaka/SiStripDigiDevice.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsSoA.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripClusterizerConditionsDevice.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/alpaka/SiStripMappingDevice.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerConditionsRecord.h"
#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"

#include "SiStripRawToClusterAlgo.h"
#include "SiStripRawToClusterHelpers.h"

// Alpaka includes
#include <alpaka/alpaka.hpp>

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

    // Containers for the condition-passing raw data
    std::vector<const FEDRawData*> raw_;
    std::vector<std::unique_ptr<FEDBuffer>> buffers_;

    std::vector<std::pair<int, int>> fedIdCh_connPairs;
    size_t fedIdCh_totalLength = 0;
    auto fillFedIdFedChBuffer(Queue& queue, const FEDRawDataCollection& rawColl);

    // Size in bytes of the condition-passing mem. buffer for FED raw
    size_t buffersValidSize_bytes_ = 0;
    // RAW unpacking mode (legacy or not)
    const bool legacyUnpacker_;
    const bool unpackBadChannels_;
    const bool doFullCorruptBufferChecks_;
    // RAW unpacking and clustering algorithm
    SiStripRawToClusterAlgo algo_;

    edm::EDGetTokenT<FEDRawDataCollection> fedRawGetToken_;
    edm::ESGetToken<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd> stripCondGetToken_;
    device::ESGetToken<SiStripClusterizerConditionsDetToFedsDevice, SiStripClusterizerConditionsDetToFedsRecord>
        stripCablCondGetToken_;
    device::ESGetToken<SiStripClusterizerConditionsDataDevice, SiStripClusterizerConditionsDataRecord>
        stripDataCondGetToken_;
    device::EDPutToken<SiStripClustersDevice> stripClustPutToken_;
    device::EDPutToken<SiStripDigiDevice> stripDigiPutToken_;

    // The unpacker and clusterizer conditions
    const SiStripClusterizerConditions* stripCond_ = nullptr;
    const SiStripClusterizerConditionsDetToFedsDevice* stripCablCond_ = nullptr;
    const SiStripClusterizerConditionsDataDevice* stripDataCond_ = nullptr;

    edm::ESWatcher<SiStripClusterizerConditionsRcd> stripCondWatcher_;
    edm::ESWatcher<SiStripClusterizerConditionsDetToFedsRecord> stripCablCondWatcher_;
    edm::ESWatcher<SiStripClusterizerConditionsDataRecord> stripDataCondWatcher_;

    // Helper functions to fill valid, condition-passing raw/buffers
    WarningSummary warnings_ = WarningSummary("", "", false);
    std::unique_ptr<FEDBuffer> fillBuffer(int fedId, const FEDRawData& rawData);
  };

  SiStripRawToCluster::SiStripRawToCluster(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        raw_(sistrip::FED_ID_MAX + 1),
        buffers_(sistrip::FED_ID_MAX + 1),
        legacyUnpacker_(iConfig.getParameter<edm::ParameterSet>("Unpacker").getParameter<bool>("LegacyUnpacker")),
        unpackBadChannels_(iConfig.getParameter<edm::ParameterSet>("Unpacker").getParameter<bool>("UnpackBadChannels")),
        doFullCorruptBufferChecks_(
            iConfig.getParameter<edm::ParameterSet>("Unpacker").getParameter<bool>("DoAllCorruptBufferChecks")),
        algo_(iConfig.getParameter<edm::ParameterSet>("Unpacker"),
              iConfig.getParameter<edm::ParameterSet>("Clusterizer")),
        fedRawGetToken_(consumes(iConfig.getParameter<edm::InputTag>("ProductLabel"))),
        stripCondGetToken_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("ConditionsLabel")})),
        stripCablCondGetToken_(
            esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("CablingConditionsLabel")})),
        stripDataCondGetToken_(esConsumes()),
        stripClustPutToken_(produces()),
        stripDigiPutToken_(produces()) {}

  void SiStripRawToCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    // Add some custom parameter description to the automatically-created ones by the addWithDefaultLabel methos
    edm::ParameterSetDescription desc;
    desc.add("ProductLabel", edm::InputTag("rawDataCollector"));
    desc.add<std::string>("ConditionsLabel", "");
    desc.add<std::string>("CablingConditionsLabel", "");

    // Unpacking parameters (from EventFilter/SiStripRawToDigi/plugins/SiStripRawToDigiModule.cc)
    // (all optional entries are to be discussed)
    edm::ParameterSetDescription unpacker;
    unpacker.addOptional<int>("AppendedBytes", 0);
    unpacker.addOptional<int>("TriggerFedId", 0);
    unpacker.add<bool>("LegacyUnpacker", false);
    unpacker.addOptional<bool>("UseDaqRegister", false);
    unpacker.addOptional<bool>("UseFedKey", false);
    unpacker.addOptional<bool>("UnpackBadChannels", false);
    unpacker.addOptional<bool>("MarkModulesOnMissingFeds", true);
    unpacker.addOptionalUntracked<int>("FedBufferDumpFreq", 0);
    unpacker.addOptionalUntracked<int>("FedEventDumpFreq", 0);
    unpacker.addOptionalUntracked<bool>("Quiet", true);
    unpacker.addOptional<bool>("UnpackCommonModeValues", false);
    unpacker.addOptional<bool>("DoAllCorruptBufferChecks", false);
    unpacker.addOptional<bool>("DoAPVEmulatorCheck", false);
    unpacker.addOptional<unsigned int>("ErrorThreshold", 7174);
    desc.add("Unpacker", unpacker);

    // Inherit all the parameters from the clusterizers (all var.s from the Clusterizer PSet)
    // 1:Algorithm, ConditionsLabel, ChannelThreshold, SeedThreshold, ClusterThreshold,
    // MaxSequentialHoles, MaxSequentialBad, MaxAdjacentBad, MaxClusterSize, RemoveApvShots,
    // setDetId, 12:clusterChargeCut
    //
    // The parameter MaxSeedStrips determines the size for pre-allocation of the clusters collection SoA
    edm::ParameterSetDescription clusterizer;
    StripClusterizerAlgorithmFactory::fillDescriptions(clusterizer);
    clusterizer.addOptional<uint32_t>("MaxSeedStrips", 150000u);
    desc.add("Clusterizer", clusterizer);

    descriptions.addWithDefaultLabel(desc);
  }

  auto SiStripRawToCluster::fillFedIdFedChBuffer(Queue& queue, const FEDRawDataCollection& rawColl) {
    // Containers for the condition-passing raw data
    std::fill(raw_.begin(), raw_.end(), nullptr);
    std::fill(buffers_.begin(), buffers_.end(), nullptr);

    std::vector<FEDChMetadata> fedChOfs_wrt_rawFedId_;
    uint32_t rawBuffFlattenSize = 0;

    // loop over good det in cabling
    for (auto idet : stripCond_->allDetIds()) {
      // it populates raw_, buffers_ with only connected fed
      auto const& det = stripCond_->findDetId(idet);
      if (!det.valid()) {
        continue;  // idet loop
      }

      // Loop over apv-pairs of det
      // int connIdx = 0;
      for (auto const conn : stripCond_->currentConnection(det)) {
        // connIdx++;
        if (!conn) {
          continue;  // conn loop
        }

        const uint16_t fedId = conn->fedId();

        // If fed id is null or connection is invalid continue
        if (!fedId || !conn->isConnected()) {
          continue;  // conn loop
        }

        if (!raw_[fedId]) {
          // pointer to the raw data in the collection
          raw_[fedId] = &rawColl.FEDData(fedId);
          rawBuffFlattenSize += raw_[fedId]->size();
        }

        // If Fed hasnt already been initialised, extract data and initialise
        sistrip::FEDBuffer* buffer = buffers_[fedId].get();
        if (!buffer) {
          buffers_[fedId] = fillBuffer(fedId, *raw_[fedId]);
          if (!buffers_[fedId]) {
            continue;
          } else {
            buffer = buffers_[fedId].get();
          }
        }
        assert(buffer);

        // Set legacy unpacking
        // buffers_[fedidx]->setLegacyMode(legacyUnpacker_);

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
        if (!buffer->channelGood(fedCh, false)) {
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

        auto diff = channel.data() - raw_[fedId]->data();
        if (diff < 0 || diff > std::numeric_limits<uint32_t>::max()) {
          // std::cout << "#diff," << diff << std::endl;
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

        // Count the total number of bytes I need for the pinned memory which is going to hold this
        // std::cout << "#bufferSize," << connIdx << "," << idet << "," << fedId << "," << buffer->bufferSize() << std::endl;
        // std::cout << "#fedChOfs," << connIdx << "," << idet << "," << fedId << "," << (int)fedCh << "," << channel.length() << "," << fedChOfs << "," << fedChOfs_wrt_rawFedId << std::endl;

        // assert(channel.length() == (channel.data()[(channel.offset()) ^ 7] + (channel.data()[(channel.offset() + 1) ^ 7] << 8)));

        // // Check that a FEDChannel built using the method effectively return the same value
        // auto fedChCheck = FEDChannel(raw_[fedId]->data()+diff, channel.offset());
        // std::cout << "#fedChChk," << fedId << "," << (int)fedCh << "," << channel.length() << "," << fedChCheck.length() << std::endl;
      }
    }

    // Create container for the buffer and mapping
    auto fedMover = std::make_unique<PortableFEDMover>(&queue, rawBuffFlattenSize, fedChOfs_wrt_rawFedId_.size());

    // Fill buffer
    fedMover->fillBuffer(raw_);

    // Fill the mapping
    fedMover->fillMapping(fedChOfs_wrt_rawFedId_);

    return fedMover;
  }

  void SiStripRawToCluster::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    // If this is the first time the module is called or the record signalled a change in conditions,
    // then load the conditions and set the cabling map

    if (stripCondWatcher_.check(iSetup)) {
      // Get the cpu con
      stripCond_ = &iSetup.getData(stripCondGetToken_);
    }

    if (stripCablCondWatcher_.check(iSetup)) {
      // Get the cabling map
      stripCablCond_ = &iSetup.getData(stripCablCondGetToken_);
      LogDebug("fedBufferBlocksRaw") << "Size of cablingMapData (bytes): "
                                     << alpaka::getExtentProduct(stripCablCond_->buffer()) * sizeof(std::byte);
    }

    if (stripDataCondWatcher_.check(iSetup)) {
      // Make the cabling and clusterizer conditions available on device
      // TO DO: automatical copy from the framework!
      stripDataCond_ = &iSetup.getData(stripDataCondGetToken_);
    }

    // Get data from the tokens (raw collection and conditions)
    const auto& rawCollection = iEvent.get(fedRawGetToken_);

    // Fill the raw_, buffers_ class members (i.e. from the connected FED, the FEDBuffers and raw pointers are set)
    std::unique_ptr<PortableFEDMover> FEDChMover = fillFedIdFedChBuffer(iEvent.queue(), rawCollection);

    // Move the DataFedAppender class to algo
    algo_.prepareUnpackCluster(iEvent.queue(), stripCablCond_, stripDataCond_, std::move(FEDChMover));
  }

  void SiStripRawToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    // Unpack the raw FED data into strip digi
    algo_.unpackStrips2(iEvent.queue());

    // Run the clusterization algorithm (ThreeThresholdAlgorithm)
    auto cluster_d = algo_.makeClusters2(iEvent.queue());

    // Get the clusters amplitudes
    auto clusterAmpls_d = algo_.getDigiAmplitudes(iEvent.queue());

    iEvent.put(stripClustPutToken_, std::move(cluster_d));
    iEvent.put(stripDigiPutToken_, std::move(clusterAmpls_d));
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

    /*
      // dump of FEDRawData to stdout
      if ( dump_ ) {
      std::stringstream ss;
      RawToDigiUnpacker::dumpRawData( fedId, input, ss );
      LogTrace(mlRawToDigi_) 
      << ss.str();
      }
      */

    return buffer;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(sistrip::SiStripRawToCluster);
