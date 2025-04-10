#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/moveToDeviceAsync.h"

#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

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

  class SiStripRawToCluster : public stream::EDProducer<> {
  public:
    SiStripRawToCluster(edm::ParameterSet const& iConfig);
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

  private:
    // Containers for the condition-passing raw data
    std::vector<const FEDRawData*> raw_;
    std::vector<std::unique_ptr<FEDBuffer>> buffers_;
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
    edm::ESGetToken<SiStripClusterizerConditionsDetToFedsHost, SiStripClusterizerConditionsDetToFedsRecord>
        stripCablCondGetToken_;
    device::ESGetToken<SiStripClusterizerConditionsDataDevice, SiStripClusterizerConditionsDataRecord>
        stripDataCondGetToken_;
    device::EDPutToken<SiStripClustersDevice> stripClustPutToken_;

    // The unpacker and clusterizer conditions
    const SiStripClusterizerConditions* stripCond_ = nullptr;
    const SiStripClusterizerConditionsDetToFedsHost* stripCablCond_ = nullptr;
    const SiStripClusterizerConditionsDataDevice* stripDataCond_ = nullptr;

    edm::ESWatcher<SiStripClusterizerConditionsRcd> stripCondWatcher_;
    edm::ESWatcher<SiStripClusterizerConditionsDetToFedsRecord> stripCablCondWatcher_;
    edm::ESWatcher<SiStripClusterizerConditionsDataRecord> stripDataCondWatcher_;

    // Helper functions to fill valid, condition-passing raw/buffers
    WarningSummary warnings_ = WarningSummary("", "", false);
    std::unique_ptr<FEDBuffer> fillBuffer(int fedId, const FEDRawData& input);
    void makeFEDbufferWithValidFEDs_(const FEDRawDataCollection& rawColl,
                                     const SiStripClusterizerConditions& conditions);
    void makeFEDbufferWithValidFEDs_4det_(uint32_t idet,
                                          const FEDRawDataCollection& rawColl,
                                          const SiStripClusterizerConditions& conditions);
  };

  SiStripRawToCluster::SiStripRawToCluster(const edm::ParameterSet& iConfig)
      : stream::EDProducer<>(iConfig),
        raw_(sistrip::NUMBER_OF_FEDS),
        buffers_(sistrip::NUMBER_OF_FEDS),
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
        stripClustPutToken_(produces()) {}

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
    edm::ParameterSetDescription clusterizer;
    StripClusterizerAlgorithmFactory::fillDescriptions(clusterizer);
    desc.add("Clusterizer", clusterizer);

    descriptions.addWithDefaultLabel(desc);
  }

  void SiStripRawToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
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
    makeFEDbufferWithValidFEDs_(rawCollection, *stripCond_);
    // raw_, buffers_ arrays are set. They are indexed by fedi := (fedID - sistrip::FED_ID_MIN)
    //  buffersValidSize_bytes_ tells the number of bytes needed to allocate

    // Create pinned host and device memory
    // fedBufferBlocksRawHost_ is made of the data and a mask (fedID) telling at each array position in data to which fedID it corresponds
    DataFedAppender fedBufferBlocksRaw(iEvent.queue(), buffersValidSize_bytes_);
    LogDebug("fedBufferBlocksRaw") << "Size of fedBufferBlocksRaw pre-allocation: " << buffersValidSize_bytes_;
    // fill the raw bytes with the buffers and store the index where a corresponding FED buffer starts
    // bytes_         = | bytes relative to a given FED id                                     | ... |
    // fedId_         = |< same size array filled with fedId_ for the whole bytes_[fed] size  >| ... |
    // chunkStartIdx_ = [0, (fedID_1 stopIdx=)fedID_2 startIdx in bytes_, (fedID_2 stopIdx=)fedID_3 startIdx in bytes_, ... ]
    // fedIDinSet_    = [2, 5, 7, ...] (list of the fedID passing the conditions)

    for (uint16_t fedi = 0; fedi < sistrip::NUMBER_OF_FEDS; ++fedi) {
      auto& buff = buffers_[fedi];
      // loop over valid buffers instead of if else is to avoid long jump in assembly
      if (!buff)
        continue;

      if (buff->checkReadoutMode()) [[likely]] {
        // Get the valid raw data
        const auto raw = raw_[fedi];
        auto fedID = fedi + sistrip::FED_ID_MIN;
        fedBufferBlocksRaw.insertFEDRawDataObj(fedID, raw);
      } else {
        // Could be moved into fillBuffer as well - understand how/if this could occur
        throw cms::Exception("RawToDigi", "Invalid readout mode for buffer ");
      }
    }

// debug
#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    fedBufferBlocksRaw_.print_Info();
#endif

    /////////////////////////////////////////////////////////////////////////////
    // @brief    Map Detector Channels to FED Data
    //           iterate over the detector in DetID/APVPair order
    //           mapping out where the data are

#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    print_SiStripClusterizerConditionsHost_(cablingMapData);
#endif
    // Prepare the external conditions (cablingMap) to be mapped onto the FEDRaw data
    auto detToFedsMap = stripCablCond_->view();

    // It contains the map between each (fedID ,fedCh, detID) and
    // 1. the raw memory location (*input, inoff, offset)
    // 2. the size (length) in the FED raw data buffer (along the fedBufferBlocksRaw_.bytes_)
    auto chanlocs_onHost = SiStripMappingHost(detToFedsMap.metadata().size(), iEvent.queue());
    LogDebug("fedBufferBlocksRaw") << "Size of chanlocs_onHost (bytes): "
                                   << alpaka::getExtentProduct(chanlocs_onHost.buffer()) * sizeof(std::byte);

    // Copy the blocks of raw FED data on the device
    auto fedBufferBlocksRaw_onDevice = cms::alpakatools::make_device_buffer<uint8_t[]>(
        iEvent.queue(), static_cast<unsigned int>(fedBufferBlocksRaw.getPreallocSize()));
    alpaka::memcpy(iEvent.queue(),
                   fedBufferBlocksRaw_onDevice,
                   fedBufferBlocksRaw.getBuffer(),
                   static_cast<unsigned int>(fedBufferBlocksRaw.getPreallocSize()));

    // -- Expand the SiStripClusterizerConditionsDetToFedsSoA to mask the fed buffers according to "good" detectors
    //    preparing the data for the unpack - In summary, make the A-B map between
    //    A - the raw FED data (a bug chunk of bytes which were copied to the device in the previous line)
    //    B - the actual fed buffers, which matters for unpacking, selecting (possibly a subset of) the detectors
    static constexpr uint32_t invalidDet = std::numeric_limits<uint32_t>::max();
    static constexpr uint16_t invalidFed = std::numeric_limits<uint16_t>::max();

    // To check. By dumping the unpacked digi, it seems that an header/trailer is present at the beginning
    //           of each channel data. If this is the case, there is room for optimization in the unpacking.
    uint32_t offset = 0;
    uint32_t n_strips = 0;
    uint32_t skippedBytes = 0;

// Loop over the allowed detID/fedID/fedCH from the conditions
#ifdef EDM_ML_DEBUG
    std::ostringstream dumpMsg("[SiStripRawToClusterAlgo::produce] Preparing strip data on host...\n");
    dumpMsg << "Pre-allocated " << detToFedsMap.metadata().size() << " elements for SiStripMappingHost\n";
    dumpMsg << " ------------ ------ Dumping loop     ------ ------------\n";
    dumpMsg << "i\tfedId\tfedCh\tlen\toff\tmy_offset\toffset\n";
#endif

    for (int i = 0; i < detToFedsMap.metadata().size(); ++i) {
      const auto detID = detToFedsMap.detid_(i);
      const auto fedID = detToFedsMap.fedid_(i);
      const auto fedCH = detToFedsMap.fedch_(i);

      const auto fedi = fedID - FED_ID_MIN;

      // Runtime check for the fedID (i.e., badly generated conditions ended up here)
      // unlikely used in https://github.com/cms-sw/cmssw/blob/7035c70e3a533533a7f8d600ff29f23579ca6add/RecoTracker/PixelTrackFitting/interface/RZLine.h#L101
      if (fedID < sistrip::FED_ID_MIN || fedID > sistrip::FED_ID_MAX) [[unlikely]] {
        // To understand whether this should stop execution or continue, skipping the ith
        throw cms::Exception("RawToDigi")
            << "Invalid fedID: " << fedID << " for detID: " << detID << " at record: " << i;
      }

      // This fedID from conditions (detToFedsMap) is within the FED raw data from collection?
      if (!fedBufferBlocksRaw.isInside(fedID)) {
        // chanlocs_onHost.view()[i] = {nullptr, 0, 0, 0, READOUT_MODE_INVALID, 0, invalidFed, 0, invalidDet};
        chanlocs_onHost.view()[i] = {invalidDet, invalidFed, invalidFed, 0, 0, 0, 0, READOUT_MODE_INVALID, 0};
      } else {
        // Get the FEDBuffer object for the current fedID
        const auto buffer = buffers_[fedi].get();

        // Get readout mode
        const FEDReadoutMode buffROMode = buffer->readoutMode();
        const FEDLegacyReadoutMode buffROModeLegacy =
            (legacyUnpacker_) ? buffer->legacyReadoutMode() : READOUT_MODE_LEGACY_INVALID;
        // Make sure EACH buffer has a readout mode supported by the module
        if (!(buffROMode >= READOUT_MODE_ZERO_SUPPRESSED_LITE10 &&
              buffROMode <= READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE &&
              buffROMode != READOUT_MODE_PROC_RAW)) {
          throw cms::Exception("RawToDigi")
              << "Unsupported readout mode: " << buffROMode << " from condition FEDID=" << fedID << " FEDCH=" << fedCH;
        }

        // ------------------------------------------------------------------------------------
        // -> at this point, the readout mode is ZERO_SUPPRESSED_* for SURE ->

        // Determine if the ZS is non-lite, retrieve the packet code, get the header len
        const bool isNonLite = fedchannelunpacker::isNonLiteZS(buffROMode, legacyUnpacker_, buffROModeLegacy);
        const uint8_t pCode = (isNonLite ? buffer->packetCode(legacyUnpacker_, fedCH) : 0);
        // The header len determines how many bytes to shift with respect to fedChannel.data(),
        // to start reading actual strip data. There is another header within the fedChannel, telling
        // the number of strips in the payload of the channel
        const int headerlen = (isNonLite ? 7 : 2);

        // Get the FEDChannel data from the buffer
        const auto& fedChannel = buffer->channel(fedCH);
        // auto data = fedChannel.data();
        auto off = fedChannel.offset();
        // The length is extracted from the first 2 bytes (assuming normal FED channel) starting from .data()
        // The len MUST be different from 0, otherwise the channel data has malformed data
        auto len = fedChannel.length();

        // To unpack data, the .data is shifted by headerLen
        // and scanned from channel.offset() + headerLength to
        // channel.offset() + channel.length(). Then headerLength < channel.length()
        if (!(headerlen <= len)) {
          // This channel data is malformed. It must be skipped
          if (edm::isDebugEnabled()) {
            edm::LogWarning("BuffCkh") << "Malformed channel data for fedID: " << fedID;
          }
          skippedBytes += len;
          // continue;
        }

        // Retrieve the position of this fedID in the pinned buffer
        auto fedOffsetInBuffer = fedBufferBlocksRaw.getOffset4FEDID(fedID);
        if (!fedOffsetInBuffer) [[unlikely]] {
          // Very bad condition, where the fedID is supposed in the buffer (isInside=true),
          // but there is no association in the map.
          if (edm::isDebugEnabled()) {
            throw cms::Exception("RawToDigi")
                << "Invalid fedID: " << fedID << " for detID: " << detID << " at record: " << i;
          }
        }

        // Coordinates for this data...
        chanlocs_onHost->detID(i) = detID;
        chanlocs_onHost->fedID(i) = fedID;
        chanlocs_onHost->fedCh(i) = fedCH;
        //
        // where this FEDChannel object is located in the pinned buffer
        auto fedChOfs_inRawBuffer = (*fedOffsetInBuffer) + (fedChannel.data() - raw_[fedi]->data());
        chanlocs_onHost->fedChOff(i) = fedChOfs_inRawBuffer;
        // -- fedchannel properties --
        chanlocs_onHost->inoff(i) = off;
        chanlocs_onHost->length(i) = len;
        chanlocs_onHost->offset(i) = offset;
        //
        // buffer-related properties (to generalize to arbitrary unpackers)
        chanlocs_onHost->readoutMode(i) = buffROMode;
        chanlocs_onHost->packetCode(i) = pCode;

        // n.b.: see comment above about the headerlen
        offset += (len - headerlen);
        n_strips += (len - headerlen);

#ifdef EDM_ML_DEBUG
        if (i % 100 == 0) {
          dumpMsg << i << "\t" << fedID << "\t" << (int)fedCH << fedi << "\t" << len << "\t" << off << "\t" << n_strips
                  << "\t" << offset << "\n";
        }
#endif
      }
    }

#ifdef EDM_ML_DEBUG
    dumpMsg << " ------------ ------ Dumping loop END ------ ------------\n";
    dumpMsg << "Skipped bytes in the unpacking: " << skippedBytes << "\n";
    LogDebug(sistrip::mlRawToCluster_) << dumpMsg.str();
#endif

    if (edm::isDebugEnabled()) {
      if (skippedBytes) {
        edm::LogWarning("BuffCkh") << "Skipped bytes in the unpacking: " << skippedBytes;
      }

      auto fedsInRaw =
          std::count(fedBufferBlocksRaw.getFedIDset()->begin(), fedBufferBlocksRaw.getFedIDset()->end(), true);
      auto actualStrips = n_strips / fedsInRaw - 2;
      LogDebug("actual strips: ") << actualStrips << " number of feds in raw: " << fedsInRaw;
    }
    //
    //
    // @brief    Map Detector Channels to FED Data - END
    /////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////
    // @brief    Run the algorithm for unpacking and clustering
    //
    //

    // Make the mapping between the unpacked FED data and strips available on the device
    auto chanlocs_onDevice = cms::alpakatools::moveToDeviceAsync(iEvent.queue(), std::move(chanlocs_onHost));

    // Prepare the StripDigiDevice var. on the host (with Clusterizer PSet)
    algo_.initialize(iEvent.queue(), n_strips);

    // Unpack the FED raw data into SiStrip digits (adc, channel, strip) (in the unpackedStrips_d_ member of the algo_)
    algo_.unpackStrips(iEvent.queue(), fedBufferBlocksRaw_onDevice.data(), chanlocs_onDevice, *stripDataCond_);

    // Make the seed mask for strip - to be used for clustering - according to noise and threshold.
    // Also, it flags non-contiguos strips (which are used in the clusterization) and it calculates exclusive prefix sum on NC-strips
    algo_.setSeedsAndMakeIndexes(iEvent.queue(), chanlocs_onDevice, *stripDataCond_);

    // Run the clusterization algorithm (ThreeThresholdAlgorithm)
    auto cluster_d = algo_.makeClusters(iEvent.queue(), chanlocs_onDevice, *stripDataCond_);

    iEvent.put(stripClustPutToken_, std::move(cluster_d));
  }

  void SiStripRawToCluster::makeFEDbufferWithValidFEDs_(const FEDRawDataCollection& rawColl,
                                                        const SiStripClusterizerConditions& conditions) {
    // Initialize the total size and variables
    // here to clear out any residual from crashinn acquire or producer not called
    buffersValidSize_bytes_ = 0;
    std::fill(raw_.begin(), raw_.end(), nullptr);
    std::fill(buffers_.begin(), buffers_.end(), nullptr);

    // loop over good det in cabling
    for (auto idet : conditions.allDetIds()) {
      // it populates raw_, buffers_ with only connected fed
      makeFEDbufferWithValidFEDs_4det_(idet, rawColl, conditions);
    }
  }

  void SiStripRawToCluster::makeFEDbufferWithValidFEDs_4det_(uint32_t idet,
                                                             const FEDRawDataCollection& rawColl,
                                                             const SiStripClusterizerConditions& conditions) {
    auto const& det = conditions.findDetId(idet);
    if (!det.valid())
      return;

    // Loop over apv-pairs of det
    for (auto const conn : conditions.currentConnection(det)) {
      if UNLIKELY (!conn)
        continue;

      // If fed id is null or connection is invalid continue
      const uint16_t fedId = conn->fedId();
      if UNLIKELY (!fedId || !conn->isConnected())
        continue;

      // If Fed hasnt already been initialised, extract data and initialise
      auto fedidx = fedId - sistrip::FED_ID_MIN;  // this is a number going from 0 to sistrip::NUMBER_OF_FEDS-1
      sistrip::FEDBuffer* buffer = buffers_[fedidx].get();
      if (!buffer) {
        const FEDRawData& rawData = rawColl.FEDData(
            fedId);  // the indexing of the data in the collection makes uses of the actual fedID (not fedID-1 !)
        raw_[fedidx] = &rawData;
        // the fillBuffer function checks for header/trailer and validate the data packet,
        // if the data is invalid, a nullptr is returned
        buffers_[fedidx] = fillBuffer(fedId, rawData);

        // Count the total number of bytes I need for the pinned memory which is going to hold this
        if (buffers_[fedidx])
          buffersValidSize_bytes_ += buffers_[fedidx]->bufferSize();
      }
    }
  }

  // Set a FEDBuffer pointer starting from the FEDRawData, pre-checking the data is valid. If not, nullptr is returned
  // note: this is the original implementation of fillBuffer. std::optional / stack cannot be used because of missing move in FEDBuffer
  //       I would have some doubts on the performance improvement from UNLIKELY/LIKELY macro, which I leave to be addressed in the PR.
  std::unique_ptr<FEDBuffer> SiStripRawToCluster::fillBuffer(int fedId, const FEDRawData& input) {
    // check FEDRawData pointer, size, and more
    const FEDBufferStatusCode st_buffer = preconstructCheckFEDBuffer(input);
    if (FEDBufferStatusCode::SUCCESS != st_buffer) [[unlikely]] {
      if (FEDBufferStatusCode::BUFFER_NULL == st_buffer) {
        warnings_.add("NULL pointer to FEDRawData for FED", fmt::format("id {0}", fedId));
        return nullptr;
      } else if (!input.size()) {
        warnings_.add("FEDRawData has zero size for FED", fmt::format("id {0}", fedId));
        return nullptr;
      } else {
        warnings_.add("Exception caught when creating FEDBuffer object for FED",
                      fmt::format("id {0}: {1}", fedId, static_cast<int>(st_buffer)));
        return nullptr;
      }
    }

    std::unique_ptr<FEDBuffer> buffer = std::make_unique<FEDBuffer>(input);
    const FEDBufferStatusCode st_chan = buffer->findChannels();

    if (FEDBufferStatusCode::SUCCESS != st_chan) [[unlikely]] {
      warnings_.add("Exception caught when creating FEDBuffer object for FED",
                    fmt::format("id {0}: {1}", fedId, static_cast<int>(st_chan)));
      return nullptr;
    }

    buffer->setLegacyMode(legacyUnpacker_);

    if ((!buffer->doChecks(false)) && (!unpackBadChannels_ || !buffer->checkNoFEOverflows())) [[unlikely]] {
      warnings_.add("Exception caught when creating FEDBuffer object for FED",
                    fmt::format("id {0}: FED Buffer check fails for FED ID {0}.", fedId));
      return nullptr;
    }
    if (doFullCorruptBufferChecks_ && !buffer->doCorruptBufferChecks()) [[unlikely]] {
      warnings_.add("Exception caught when creating FEDBuffer object for FED",
                    fmt::format("id {0}: FED corrupt buffer check fails for FED ID {0}.", fedId));
      return nullptr;
    }

    return buffer;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(sistrip::SiStripRawToCluster);
