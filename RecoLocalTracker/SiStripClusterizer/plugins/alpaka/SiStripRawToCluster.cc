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

// Alpaka includes
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  using namespace ::sistrip;

  namespace {
    // Set a FEDBuffer pointer starting from the FEDRawData, pre-checking the data is valid. If not, nullptr is returned
    // note: this is the original implementation of fillBuffer. std::optional could be used instead of nullptr.
    //       I would have some doubts on the performance improvement from UNLIKELY/LIKELY macro, which I leave to be addressed in the PR.
    std::unique_ptr<sistrip::FEDBuffer> fillBuffer(int fedId, const FEDRawData& rawData) {
      std::unique_ptr<sistrip::FEDBuffer> buffer;

      // Check on FEDRawData pointer
      const auto st_buffer = sistrip::preconstructCheckFEDBuffer(rawData);
      if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_buffer) {
        LogDebug(sistrip::mlRawToCluster_)
            << "[ClustersFromRawProducer::" << __func__ << "]" << st_buffer << " for FED ID " << fedId;
        return buffer;
      }

      buffer = std::make_unique<sistrip::FEDBuffer>(rawData);
      const auto st_chan = buffer->findChannels();

      if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_chan) {
        LogDebug(sistrip::mlRawToCluster_)
            << "Exception caught when creating FEDBuffer object for FED " << fedId << ": " << st_chan;
        buffer.reset();
        return buffer;
      }

      if UNLIKELY (!buffer->doChecks(false)) {
        LogDebug(sistrip::mlRawToCluster_)
            << "Exception caught when creating FEDBuffer object for FED " << fedId << ": FED Buffer check fails";
        buffer.reset();
        return buffer;
      }

      return buffer;
    }
  }  // namespace

  class SiStripRawToCluster : public stream::EDProducer<> {
  public:
    SiStripRawToCluster(edm::ParameterSet const& iConfig);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

  private:
    // inputs
    edm::EDGetTokenT<FEDRawDataCollection> fedRawDataGetToken_;
    edm::ESGetToken<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd> siStrCondGetToken_;
    edm::ESGetToken<SiStripClusterizerConditionsHost, SiStripClusterizerConditionsRecord> siStCabCondGetToken_;

    // outputs
    device::EDPutToken<SiStripClustersDevice> siStrClustsPutToken_;

    // RAW data unpacking
    // FEDRawData locations and FEDBuffers passing the conditions
    std::vector<const FEDRawData*> raw_;
    std::vector<std::unique_ptr<sistrip::FEDBuffer>> buffers_;
    // Total size in bytes to be copied on device (conditions applied)
    size_t buffersValidSize_bytes_{0};
    // It populates the raw_, buffers_ members with valid (connected detectors with non-null fed buffers) FED data
    void makeFEDbufferWithValidFEDs_(const FEDRawDataCollection& rawColl,
                                     const SiStripClusterizerConditions& conditions);
    // It populates the raw_, buffers_ members with idet data masking valid FED data (connected detectors with non-null fed buffers)
    void makeFEDbufferWithValidFEDs_4det_(uint32_t idet,
                                          const FEDRawDataCollection& rawColl,
                                          const SiStripClusterizerConditions& conditions);

    // FED unpacker and clusterizer algorithms
    SiStripRawToClusterAlgo algo_;

    bool legacy_ = false;  // legacy unpacking mode, for the future

// Debug functions
#ifdef EDM_ML_DEBUG
    void print_SiStripClusterizerConditions_(SiStripClusterizerConditions const& conditions);
    void print_SiStripClusterizerConditionsHost_(SiStripClusterizerConditionsHost const& conditions);
#endif
  };

  SiStripRawToCluster::SiStripRawToCluster(const edm::ParameterSet& iConfig)
      : EDProducer<>(iConfig),
        fedRawDataGetToken_(consumes(iConfig.getParameter<edm::InputTag>("ProductLabel"))),
        siStrCondGetToken_(esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("ConditionsLabel")})),
        siStCabCondGetToken_(
            esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("CablingConditionsLabel")})),
        siStrClustsPutToken_(produces()),
        raw_(sistrip::NUMBER_OF_FEDS),
        buffers_(sistrip::NUMBER_OF_FEDS),
        algo_(iConfig.getParameter<edm::ParameterSet>("Clusterizer"), iConfig.getParameter<bool>("LegacyUnpacker")) {
    // legacy unpacking shall be supported ? (under discussion)
  }

  void SiStripRawToCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    // Add some custom parameter description to the automatically-created ones by the addWithDefaultLabel methos
    edm::ParameterSetDescription desc;
    desc.add("ProductLabel", edm::InputTag("rawDataCollector"));
    desc.add<std::string>("ConditionsLabel", "");
    desc.add<std::string>("CablingConditionsLabel", "");

    // Raw algorithms
    // edm::ParameterSetDescription rawAlgoDescr;
    // rawAlgoDescr.add<bool>("LegacyUnpacker", false);
    // rawAlgoDescr.add<bool>("Use10bitsTruncation", false);
    desc.add<bool>("LegacyUnpacker", false);

    // Inherit all the parameters from the clusterizers (all var.s from the Clusterizer PSet)
    edm::ParameterSetDescription clusterizer;
    StripClusterizerAlgorithmFactory::fillDescriptions(clusterizer);
    desc.add("Clusterizer", clusterizer);

    descriptions.addWithDefaultLabel(desc);
  }

  void SiStripRawToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    // Get data from the tokens (raw collection and conditions)
    const auto& rawCollection = iEvent.get(fedRawDataGetToken_);
    const auto& validFEDsConditions = iSetup.getData(siStrCondGetToken_);

    // Get the cabling map
    const auto& cablingMapData = iSetup.getData(siStCabCondGetToken_);

#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    print_SiStripClusterizerConditions_(validFEDsConditions);
    print_SiStripClusterizerConditionsHost_(cablingMapData);
#endif

    // Fill the raw_, buffers_ class members (i.e. from the connected FED, the FEDBuffers and raw pointers are set)
    makeFEDbufferWithValidFEDs_(rawCollection, validFEDsConditions);
    // raw_, buffers_ arrays are set. They are indexed by fedi := (fedID - sistrip::FED_ID_MIN)
    //  buffersValidSize_bytes_ tells the number of bytes needed to allocate

    // Create pinned host and device memory
    // fedBufferBlocksRawHost_ is made of the data and a mask (fedID) telling at each array position in data to which fedID it corresponds
    DataFedAppender fedBufferBlocksRaw_(iEvent.queue(), buffersValidSize_bytes_);
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

      if (!buff->checkReadoutMode()) {
        // Could be moved into fillBuffer as well - understand how/if this could occur
        throw cms::Exception("RawToDigi", "Invalid readout mode for buffer ");
      }

      // Get the valid raw data
      const auto raw = raw_[fedi];
      auto fedID = fedi + sistrip::FED_ID_MIN;
      fedBufferBlocksRaw_.insertFEDRawDataObj(fedID, raw);
    }

// debug
#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    fedBufferBlocksRaw_.print_Info();
#endif

    /////////////////////////////////////////////////////////////////////////////
    // @brief    Map Detector Channels to FED Data
    //           iterate over the detector in DetID/APVPair order
    //           mapping out where the data are

    // Prepare the external conditions (cablingMap) to be mapped onto the FEDRaw data
    auto detToFedsMap = cablingMapData.view<SiStripClusterizerConditionsDetToFedsSoA>();

    // It contains the map between each (fedID ,fedCh, detID) and
    // 1. the raw memory location (*input, inoff, offset)
    // 2. the size (length) in the FED raw data buffer (along the fedBufferBlocksRaw_.bytes_)
    auto chanlocs_onHost = SiStripMappingHost(detToFedsMap.metadata().size(), iEvent.queue());

    // This object contains the addresses of the input raw data for channel
    auto rawPointerAddresses_onHost =
        cms::alpakatools::make_host_buffer<const uint8_t*[]>(iEvent.queue(), detToFedsMap.metadata().size());

    // Copy the blocks of raw FED data on the device
    auto fedBufferBlocksRaw_onDevice = cms::alpakatools::make_device_buffer<uint8_t[]>(
        iEvent.queue(), static_cast<unsigned int>(fedBufferBlocksRaw_.size()));
    alpaka::memcpy(iEvent.queue(),
                   fedBufferBlocksRaw_onDevice,
                   fedBufferBlocksRaw_.getData(),
                   static_cast<unsigned int>(fedBufferBlocksRaw_.size()));

    // -- Expand the SiStripClusterizerConditionsDetToFedsSoA to mask the fed buffers according to "good" detectors
    //    preparing the data for the unpack - In summary, make the A-B map between
    //    A - the raw FED data (a bug chunk of bytes which were copied to the device in the previous line)
    //    B - the actual fed buffers, which matters for unpacking, selecting (possibly a subset of) the detectors
    static constexpr uint32_t invalidDet = std::numeric_limits<uint32_t>::max();
    static constexpr uint16_t invalidFed = std::numeric_limits<uint16_t>::max();

    // To check. By dumping the unpacked digi, it seems that an header/trailer is present at the beginning
    //           of each channel data. If this is the case, there is room for optimization in the unpacking.
    uint32_t offset = 0;
    int n_strips = 0;

    // Loop over the allowed detID/fedID/fedCH from the conditions
#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
    LogDebug("SiStripPrtArit") << "i\tfedId\tfedCh\tfedi\tlen\toff\tmy_offset\toffset\n";
#endif

    for (int i = 0; i < detToFedsMap.metadata().size(); ++i) {
      const auto detID = detToFedsMap.detid_(i);
      const auto fedID = detToFedsMap.fedid_(i);
      const auto fedCH = detToFedsMap.fedch_(i);
      const auto fedi = fedID - sistrip::FED_ID_MIN;

      if (fedBufferBlocksRaw_.isInside(fedID)) {
        const auto buffer = buffers_[fedi].get();

        /// extract readout mode
        const sistrip::FEDReadoutMode bufferReadoutMode = buffer->readoutMode();
        const sistrip::FEDLegacyReadoutMode lmode =
            (legacy_) ? buffer->legacyReadoutMode() : sistrip::READOUT_MODE_LEGACY_INVALID;
        const bool isNonLite = fedchannelunpacker::isNonLiteZS(bufferReadoutMode, legacy_, lmode);
        const uint8_t pCode = (isNonLite ? buffer->packetCode(legacy_, fedCH) : 0);

        // #ifdef EDM_ML_DEBUG
        // if (isNonLite) {
        //   LogDebug("SiStripRawToDigi") << "Non-lite zero-suppressed mode. Packet code=0x" << std::hex << uint16_t(pCode) << std::dec;
        // }
        // #endif

        if (!(bufferReadoutMode >= READOUT_MODE_ZERO_SUPPRESSED_LITE10 &&
              bufferReadoutMode <= READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE &&
              bufferReadoutMode != READOUT_MODE_PROC_RAW)) {
          throw cms::Exception("RawToDigi") << "Unsupported readout mode: " << bufferReadoutMode
                                            << " from condition FEDID=" << fedID << " FEDCH=" << fedCH;
        }

        // preparing ground for legacy unpacking...
        int headerlen = (isNonLite ? 7 : 2);
        if ((!legacy_) ? bufferReadoutMode == READOUT_MODE_PREMIX_RAW : lmode == READOUT_MODE_LEGACY_PREMIX_RAW) {
          headerlen = 7;
        }

        const auto& fedChannel = buffer->channel(fedCH);
        auto data = fedChannel.data();
        auto len = fedChannel.length();
        auto off = fedChannel.offset();

        // What about len==0 case?
        assert(len >= headerlen);

        // the input will be overridden with pointers to device memory after the memcpy to the device
        chanlocs_onHost->input(i) = data;
        chanlocs_onHost->inoff(i) = off;
        chanlocs_onHost->length(i) = len;
        // -- fedchannel properties --
        // global offset for the FEDChannel in the rawFEDBuffer copied on the device
        chanlocs_onHost->offset(i) = offset;
        // buffer-related properties (to generalize to arbitrary unpackers)
        chanlocs_onHost->readoutMode(i) = bufferReadoutMode;
        chanlocs_onHost->packetCode(i) = pCode;
        chanlocs_onHost->fedID(i) = fedID;
        chanlocs_onHost->fedCh(i) = fedCH;
        chanlocs_onHost->detID(i) = detID;

        // ##TODO Better comments - fedBufferBlocksRaw_.getOffset(fedID) this is just an offset - same in host/dev mem
        auto fedChOfs_inRawBuffer = fedBufferBlocksRaw_.getOffset(fedID) + (fedChannel.data() - raw_[fedi]->data());
        // this is the offset between the pointer where the channel data begins and the position of the data as a whole start (this difference is the same in host/device, and this is calculated on host - since both things are in host memory already)

        // For debugging on host, use fedBufferBlocksRaw_.getData().data() instead
        rawPointerAddresses_onHost[i] = fedBufferBlocksRaw_onDevice.data() + fedChOfs_inRawBuffer;

        // n.b.: see comment above about the headerlen
        offset += (len - headerlen);
        n_strips += (len - headerlen);

#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
        if (i % 100 == 0)
          LogDebug("SiStripPrtArit") << i << "\t" << fedID << "\t" << (int)fedCH << fedi << "\t" << len << "\t" << off
                                     << "\t" << my_offset << "\t" << offset << "\n";
#endif
      } else {
        chanlocs_onHost.view()[i] = {nullptr, 0, 0, 0, READOUT_MODE_INVALID, 0, invalidFed, 0, invalidDet};
        rawPointerAddresses_onHost[i] = nullptr;
      }
    }
    //
    //
    // @brief    Map Detector Channels to FED Data - END
    /////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////
    // @brief    Run the algorithm for unpacking and clustering
    //
    //
    // Make the cabling and clusterizer conditions available on device
    // TO DO: automatical copy from the framework!
    auto clusterizerConditions_onDevice = SiStripClusterizerConditionsDevice(cablingMapData.sizes(), iEvent.queue());
    alpaka::memcpy(iEvent.queue(), clusterizerConditions_onDevice.buffer(), cablingMapData.const_buffer());

    // Make the mapping between the unpacked FED data and strips available on the device
    auto chanlocs_onDevice = cms::alpakatools::moveToDeviceAsync(iEvent.queue(), std::move(chanlocs_onHost));

    // override the .input() member of the chanlocs_onDevice - i.e., containing the device pointers of the FED raw data -
    auto moduleStartFirstElement = cms::alpakatools::make_device_view(
        iEvent.queue(), chanlocs_onDevice.view().input(), detToFedsMap.metadata().size());
    alpaka::memcpy(iEvent.queue(), moduleStartFirstElement, rawPointerAddresses_onHost);

    // Prepare the StripDigiDevice var. on the host (with Clusterizer PSet)
    algo_.initialize(iEvent.queue(), n_strips);

    // Unpack the FED raw data into SiStrip digits (adc, channel, strip) (in the unpackedStrips_d_ member of the algo_)
    algo_.unpackStrips(iEvent.queue(), chanlocs_onDevice, clusterizerConditions_onDevice);

    // Make the seed mask for strip - to be used for clustering - according to noise and threshold.
    // Also, it flags non-contiguos strips (which are used in the clusterization) and it calculates exclusive prefix sum on NC-strips
    algo_.setSeedsAndMakeIndexes(iEvent.queue(), chanlocs_onDevice, clusterizerConditions_onDevice);

    // Run the clusterization algorithm (ThreeThresholdAlgorithm)
    algo_.makeClusters(iEvent.queue(), chanlocs_onDevice, clusterizerConditions_onDevice);

    iEvent.emplace(siStrClustsPutToken_, algo_.getClustersDevice());
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

// Debug functions
#ifdef EDM_ML_DEBUG
  void SiStripRawToCluster::print_SiStripClusterizerConditions_(SiStripClusterizerConditions const& conditions) {
    std::vector<unsigned int> detectors;
    std::vector<unsigned int> fedIds;
    // loop over good det in cabling
    for (auto idet : conditions.allDetIds()) {
      auto const& det = conditions.findDetId(idet);
      if (!det.valid())
        return;

      detectors.emplace_back(idet);

      for (auto const conn : conditions.currentConnection(det)) {
        const uint16_t fedId = conn->fedId();
        if UNLIKELY (!fedId || !conn->isConnected())
          continue;

        fedIds.emplace_back(fedId);
      }
    }  // end loop over dets

    LogDebug("SiStripConditions") << "There are " << detectors.size() << " detectors ID which are valid.\n";
    for (size_t i = 0; i < detectors.size(); ++i) {
      if (i % 100 == 0)
        LogDebug("SiStripConditions") << "  " << i << ":" << detectors[i];
    }
    LogDebug("SiStripConditions") << "There are " << fedIds.size()
                                  << " fedIDs which are attached to connected FEDs. These are:\n\t";
    for (size_t i = 0; i < fedIds.size(); ++i) {
      if (i % 100 == 0)
        LogDebug("SiStripConditions") << "  " << i << ":" << fedIds[i];
    }
    LogDebug("SiStripConditions") << "\n";
  }

  void SiStripRawToCluster::print_SiStripClusterizerConditionsHost_(SiStripClusterizerConditionsHost const& conditions) {
    auto SiStripClusterizerConditionsDetToFedsSoA_view = conditions.view<SiStripClusterizerConditionsDetToFedsSoA>();
    auto SiStripClusterizerConditionsData_fedchSoA_view = conditions.view<SiStripClusterizerConditionsData_fedchSoA>();
    auto SiStripClusterizerConditionsData_stripSoA_view = conditions.view<SiStripClusterizerConditionsData_stripSoA>();
    auto SiStripClusterizerConditionsData_apvSoA_view = conditions.view<SiStripClusterizerConditionsData_apvSoA>();

    // alpaka::wait(queue);

    LogDebug("SiStripConditions") << "From conditions, there are "
                                  << SiStripClusterizerConditionsDetToFedsSoA_view.metadata().size() << " entries\n";
    LogDebug("SiStripConditions") << "-- SiStripClusterizerConditionsDetToFedsSoA_view --\n"
                                  << "i \t detid \t ipair \t fedid \t fedch\n";

    for (int i = 0; i < SiStripClusterizerConditionsDetToFedsSoA_view.metadata().size(); ++i) {
      if ((i < 1000 || i > (SiStripClusterizerConditionsDetToFedsSoA_view.metadata().size() - 1000))) {
        LogDebug("SiStripConditions") << i << "\t" << SiStripClusterizerConditionsDetToFedsSoA_view.detid_(i) << "\t"
                                      << SiStripClusterizerConditionsDetToFedsSoA_view.ipair_(i) << "\t"
                                      << SiStripClusterizerConditionsDetToFedsSoA_view.fedid_(i) << "\t"
                                      << (int)(SiStripClusterizerConditionsDetToFedsSoA_view.fedch_(i)) << "\n";
      }
    }

    LogDebug("SiStripConditions") << "----------"
                                  << "\n";
    // return;
    LogDebug("SiStripConditions") << "-- SiStripClusterizerConditionsData_fedchSoA_view --\n"
                                  << "i \t detID \t iPair \t invthick\n";
    for (int i = 0; i < SiStripClusterizerConditionsData_fedchSoA_view.metadata().size(); ++i) {
      if ((i < 1000 || i > (SiStripClusterizerConditionsData_fedchSoA_view.metadata().size() - 1000))) {
        LogDebug("SiStripConditions") << i << "\t" << SiStripClusterizerConditionsData_fedchSoA_view.detID_(i) << "\t"
                                      << SiStripClusterizerConditionsData_fedchSoA_view.iPair_(i) << "\t"
                                      << SiStripClusterizerConditionsData_fedchSoA_view.invthick_(i) << "\n";
      }
    }

    LogDebug("SiStripConditions") << "----------"
                                  << "\n";

    LogDebug("SiStripConditions") << "-- SiStripClusterizerConditionsData_stripSoA_view --\n"
                                  << "i \t noise \n";
    for (int i = 0; i < SiStripClusterizerConditionsData_stripSoA_view.metadata().size(); ++i) {
      if ((i < 1000 || i > (SiStripClusterizerConditionsData_stripSoA_view.metadata().size() - 1000))) {
        LogDebug("SiStripConditions") << i << "\t" << SiStripClusterizerConditionsData_stripSoA_view.noise_(i) << "\n";
      }
    }

    LogDebug("SiStripConditions") << "----------"
                                  << "\n";

    LogDebug("SiStripConditions") << "-- SiStripClusterizerConditionsData_apvSoA_view --\n"
                                  << "i \t gain \n";
    for (int i = 0; i < SiStripClusterizerConditionsData_apvSoA_view.metadata().size(); ++i) {
      if ((i < 1000 || i > (SiStripClusterizerConditionsData_apvSoA_view.metadata().size() - 1000))) {
        LogDebug("SiStripConditions") << i << "\t" << SiStripClusterizerConditionsData_apvSoA_view.gain_(i) << "\n";
      }
    }
  }
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(sistrip::SiStripRawToCluster);
