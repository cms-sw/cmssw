#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripClusterSoA/interface/alpaka/SiStripClustersDevice.h"

#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsHost.h"
#include "CondFormats/SiStripObjects/interface/alpaka/SiStripMappingDevice.h"

#include "SiStripRawToClusterAlgo.h"

// Alpaka includes
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
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
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class SiStripRawToCluster : public stream::EDProducer<> {
  public:
    SiStripRawToCluster(edm::ParameterSet const& iConfig);
    // ~SiStripRawToCluster() override = default; // no reason to include if the default is called, but probably I am implicitly assuming a final class

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    // void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

  private:
    // inputs
    edm::EDGetTokenT<FEDRawDataCollection> fedRawDataGetToken_;
    edm::ESGetToken<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd> siStripConditionsGetToken_;
    edm::ESGetToken<SiStripClusterizerConditionsHost, SiStripClusterizerConditionsRcd> siStripCablingConditionsGetToken_;

    // outputs
    device::EDPutToken<sistrip::SiStripClustersDevice> siStripClustersDevicePutToken_;
    // edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClustersSetVecPutToken_;

    // RAW data unpacking
    // Container for the FEDRawData from the input FEDRawDataCollection
    std::vector<const FEDRawData*> raw_;
    // Pointers to the FEDBuffers for each FED channel from the input FEDRawDataCollection
    std::vector<std::unique_ptr<sistrip::FEDBuffer>> buffers_;
    // Total size in bytes of really necessary FEDBuffers (really necessary = after conditions are applied)
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

    // Debug functions
    // #ifdef EDM_ML_DEBUG
    void print_SiStripClusterizerConditions_(SiStripClusterizerConditions const& conditions);
    void print_SiStripClusterizerConditionsHost_(SiStripClusterizerConditionsHost const& conditions);
    void print_SiStripDataCompare_(Queue& queue,
                                   SiStripMappingHost const& chanlocs,
                                   SiStripClusterizerConditionsHost const& cablingMapData,
                                   const int n_strips,
                                   bool extendedPrint);
    // #endif
  };

  SiStripRawToCluster::SiStripRawToCluster(const edm::ParameterSet& iConfig)
      : EDProducer<>(iConfig),
        raw_(sistrip::NUMBER_OF_FEDS),
        buffers_(sistrip::NUMBER_OF_FEDS),
        algo_(iConfig.getParameter<edm::ParameterSet>("Clusterizer")) {
    fedRawDataGetToken_ = consumes(iConfig.getParameter<edm::InputTag>("ProductLabel"));
    siStripConditionsGetToken_ = esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("ConditionsLabel")});
    siStripCablingConditionsGetToken_ =
        esConsumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("CablingConditionsLabel")});
    // siStripClustersSetVecPutToken_ = produces();
    siStripClustersDevicePutToken_ = produces();
  }

  void SiStripRawToCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    // Add some custom parameter description to the automatically-created ones by the addWithDefaultLabel methos
    edm::ParameterSetDescription desc;
    desc.add("ProductLabel", edm::InputTag("rawDataCollector"));
    desc.add<std::string>("ConditionsLabel", "");
    desc.add<std::string>("CablingConditionsLabel", "");

    // Inherit all the parameters from the clusterizers (all var.s from the Clusterizer PSet)
    edm::ParameterSetDescription clusterizer;
    StripClusterizerAlgorithmFactory::fillDescriptions(clusterizer);
    desc.add("Clusterizer", clusterizer);

    descriptions.addWithDefaultLabel(desc);
  }

  void SiStripRawToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    // Get data from the tokens (raw from fed and conditions)
    const auto& rawCollection = iEvent.get(fedRawDataGetToken_);
    const auto& validFEDsConditions = iSetup.getData(siStripConditionsGetToken_);

    // Get the cabling map
    const auto& cablingMapData = iSetup.getData(siStripCablingConditionsGetToken_);

#ifdef EDM_ML_DEBUG
    print_SiStripClusterizerConditions_(validFEDsConditions);
    print_SiStripClusterizerConditionsHost_(cablingMapData);
#endif

    // Fill the raw_, buffers_ class members (i.e., from connected FED the FEDBuffers (and raw pointers) are populated)
    // [more precisely, I have the pointers of the raw_ and buffers_ pointing to valid data from the rawCollection]
    makeFEDbufferWithValidFEDs_(rawCollection, validFEDsConditions);
    // raw_, buffers_ (these arrays have the index fedinx = fedId - sistrip::FED_ID_MIN), also the minimum size in bytes required for cp into device memory is in buffersValidSize_bytes_

    // Create pinned host and device memory
    // fedBufferBlocksRawHost_ is made of the data and a mask (fedID) telling at each array position in data to which fedID it corresponds
    DataFedAppender fedBufferBlocksRaw_(iEvent.queue(), buffersValidSize_bytes_);
    // fill the raw bytes with the buffers and store the index where a corresponding FED buffer starts
    // bytes_         = | bytes relative to a given FED id                                     | ... |
    // fedId_         = |< same size array filled with fedId_ for the whole bytes_[fed] size  >| ... |
    // chunkStartIdx_ = [0, (fedID_1 stopIdx=)fedID_2 startIdx in bytes_, (fedID_2 stopIdx=)fedID_3 startIdx in bytes_, ... ]
    // fedIDinSet_    = [2, 5, 7, ...] (list of the fedID passing the conditions)
    size_t offset_withinFedBufferBlocksRaw = 0;
    sistrip::FEDReadoutMode mode = sistrip::READOUT_MODE_INVALID;

    for (uint16_t fedi = 0; fedi < sistrip::NUMBER_OF_FEDS; ++fedi) {
      auto& buff = buffers_[fedi];
      // loop over valid buffers instead of if else is to avoid long jump in assembly
      if (!buff)
        continue;

      // Get the valid raw data
      const auto raw = raw_[fedi];
      auto fedID = fedi + sistrip::FED_ID_MIN;
      fedBufferBlocksRaw_.insertFEDRawDataObj(fedID, raw);

      // Make sure the readout mode is for all non-null buffers set to the same value (later this is checked to be either READOUT_MODE_ZERO_SUPPRESSED or READOUT_MODE_ZERO_SUPPRESSED_LITE10)
      if (offset_withinFedBufferBlocksRaw == 0) {
        mode = buff->readoutMode();
        if (mode == sistrip::READOUT_MODE_INVALID)
          throw cms::Exception("RawToDigi", "Invalid readout mode for the first \"supposedly valid\" buffer");
      } else if (buff->readoutMode() != mode) {
        throw cms::Exception("RawToDigi") << "Inconsistent readout mode for fedID " << fedID
                                          << " where readout mode is " << buff->readoutMode() << " != " << mode;
      }

      // Update the position of the pointer
      offset_withinFedBufferBlocksRaw += raw->size();
    }

// debug
#ifdef EDM_ML_DEBUG
    fedBufferBlocksRaw_.print_Info();
#endif

    // Verify Readout Mode
    if ((mode != sistrip::READOUT_MODE_ZERO_SUPPRESSED) && (mode != sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE10))
      throw cms::Exception("RawToDigi") << "Unsupported readout mode: " << mode;

    /////////////////////////////////////////////////////////////////////////////
    // @brief    Map Detector Channels to FED Data
    //           iterate over the detector in DetID/APVPair order
    //           mapping out where the data are

    // Prepare the external conditions (cablingMap) to be mapped onto the FEDRaw data
    auto detToFedsMap = cablingMapData.view<SiStripClusterizerConditionsDetToFedsSoA>();

    // It contains the map between each (fedID ,fedCh, detID) and the raw memory location (*input, inoff, offset) and size (length) in the FED raw data buffer -  inside fedBufferBlocksRaw_.bytes_
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
    //    preparing the data for the unpack - THIS part essentially generates the A-B map between
    //    A - the raw FED data (a bug chunk of bytes which were copied to the GPU in the previous line)
    //    B - the actual fed buffers, which matters for unpacking, selecting (possibly a subset of) the detectors
    static constexpr uint32_t invalidDet = std::numeric_limits<uint32_t>::max();
    static constexpr uint16_t invalidFed = std::numeric_limits<uint16_t>::max();
    // static constexpr uint16_t invalidStrip = std::numeric_limits<uint16_t>::max();
    const uint16_t headerlen = (mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED ? 7 : 2);
    uint32_t offset = 0;

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

        const auto& fedChannel = buffer->channel(fedCH);

        auto len = fedChannel.length();
        auto off = fedChannel.offset();

        assert(len >= headerlen || len == 0);
        assert((len & 0xF000) == false);

        if (len >= headerlen) {
          len -= headerlen;
          off += headerlen;
        }

        //@pietroGru saving the channel location (association between detector channels and fed id) in chanlocs
        // chanlocs.view()[i] = {fedCh_object.data(), off, offset, len, fedId, fedCh, detp.detid_()};
        chanlocs_onHost->input(i) =
            fedChannel
                .data();  // storing this pointer is used for debugging the algorithm on the host, its cost is negligible
        chanlocs_onHost->inoff(i) =
            off;  // internal offset within the fedchannel for the strip data (removed already of the header)
        chanlocs_onHost->offset(i) =
            offset;                        // global offset for this strip data in the rawFEDBuffer copied on the device
        chanlocs_onHost->length(i) = len;  // length of the fedchannel data (without the header already)
        chanlocs_onHost->fedID(i) = fedID;  // id of the fed
        chanlocs_onHost->fedCh(i) = fedCH;  // fed channel
        chanlocs_onHost->detID(i) = detID;  // detector ID
        //@pietroGru calculate the correct pointer inside the fedRawDataGPU where the detector's channel begins
        auto my_offset =
            fedBufferBlocksRaw_.getOffset(fedID)  // this is just an offset - same in host/dev mem
            +
            (fedChannel.data() -
             raw_[fedi]
                 ->data());  // this is the offset between the pointer where the channel data begins and the position of the data as a whole start (this difference is the same in host/device, and this is calculated on host - since both things are in host memory already)

        rawPointerAddresses_onHost[i] =
            // fedBufferBlocksRaw_.getData().data()       // this pointer is in the host memory (used in debug only)
            fedBufferBlocksRaw_onDevice.data()  // this pointer is in the device memory
            + my_offset;                        // this is the offset I need to add to reach the FEDChannel object

        offset += len;

#if defined(EDM_ML_DEBUG) && defined(SUPERDETAILS)
        if (i % 100 == 0)
          LogDebug("SiStripPrtArit") << i << "\t" << fedID << "\t" << (int)fedCH << fedi << "\t" << len << "\t" << off
                                     << "\t" << my_offset << "\t" << offset << "\n";
#endif
      } else {
        chanlocs_onHost.view()[i] = {nullptr, 0, 0, 0, invalidFed, 0, invalidDet};
        rawPointerAddresses_onHost[i] = nullptr;
      }
    }

#ifdef EDM_ML_DEBUG
    alpaka::wait(iEvent.queue());
#endif

    //
    //
    // @brief    Map Detector Channels to FED Data - END
    /////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////
    // @brief    Run the algorithm for unpacking and clustering
    //
    //
    const int n_strips = offset;

    // Make the cabling and clusterizer conditions available on device
    auto clusterizerConditions_onDevice = SiStripClusterizerConditionsDevice(cablingMapData.sizes(), iEvent.queue());
    alpaka::memcpy(iEvent.queue(), clusterizerConditions_onDevice.buffer(), cablingMapData.const_buffer());

    // Make the mapping between the unpacked FED data and strips available on the device
    auto chanlocs_onDevice = SiStripMappingDevice(chanlocs_onHost->metadata().size(), iEvent.queue());
    alpaka::memcpy(iEvent.queue(), chanlocs_onDevice.buffer(), chanlocs_onHost.const_buffer());
    // override the .input() member of the chanlocs_onDevice - i.e., containing the device pointers of the FED raw data -
    auto moduleStartFirstElement = cms::alpakatools::make_device_view(
        iEvent.queue(), chanlocs_onDevice.view().input(), detToFedsMap.metadata().size());
    alpaka::memcpy(iEvent.queue(), moduleStartFirstElement, rawPointerAddresses_onHost);

    // Prepare the StripDigiDevice var. on the host, moving the variables (channelThreshold, seedThreshold, clusterThresholdSquared, maxSequentialHoles, maxSequentialBad, maxAdjacentBad, maxClusterSize, minGoodCharge, clusterSizeLimit)
    // and reserving the n_strips
    algo_.initialize(iEvent.queue(), n_strips);

    // Unpack the FED raw data into SiStrip digits (adc, channel, strip) (in the unpackedStrips_d_ member of the algo_)
    algo_.unpackStrips(iEvent.queue(), chanlocs_onDevice, clusterizerConditions_onDevice);

#ifdef EDM_ML_DEBUG
    print_SiStripDataCompare_(iEvent.queue(), chanlocs_onHost, cablingMapData, n_strips, true);
#endif

    // Make the seed mask for strip - to be used for clustering - according to noise and threshold.
    // Also, it flags non-contiguos strips (which are used in the clusterization) and calculates exclusive sum on NC-strips
    algo_.setSeedsAndMakeIndexes(iEvent.queue(), chanlocs_onDevice, clusterizerConditions_onDevice);

    // Run the clusterization algorithm
    algo_.makeClusters(iEvent.queue(), chanlocs_onDevice, clusterizerConditions_onDevice);

    iEvent.emplace(siStripClustersDevicePutToken_, algo_.getClustersDevice());
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
      makeFEDbufferWithValidFEDs_4det_(
          idet, rawColl, conditions);  // it populates raw_, buffers_ with only connected fed
    }  // end loop over dets
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
    }  // end loop over conn
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// Debug functions
#ifdef EDM_ML_DEBUG
namespace ALPAKA_ACCELERATOR_NAMESPACE {
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

  void SiStripRawToCluster::print_SiStripDataCompare_(Queue& queue,
                                                      SiStripMappingHost const& chanlocs,
                                                      SiStripClusterizerConditionsHost const& cablingMapData,
                                                      const int n_strips,
                                                      bool extendedPrint) {
    // Unpack on the host
    auto unpackedStrips_host = StripDigiHost(n_strips, queue);
    if (extendedPrint) {
      LogDebug("SiStripConditions") << "chan"
                                    << "\t"
                                    << "fedid"
                                    << "\t"
                                    << "fedch"
                                    << "\t"
                                    << "idx"
                                    << "\t"
                                    << "ipair"
                                    << "\t"
                                    << "ipoff"
                                    << "\t"
                                    << "aoff"
                                    << "\t"
                                    << "choff"
                                    << "\t"
                                    << "len"
                                    << "\t"
                                    << "(choff++)^7"
                                    << "\n";
    }

    int totalentries = 0;
    for (auto chan = 0; chan < chanlocs->metadata().size(); ++chan) {
      const auto fedID = chanlocs->fedID(chan);
      const auto fedCH = chanlocs->fedCh(chan);

      const auto idx = (fedID - sistrip::FED_ID_MIN) * sistrip::FEDCH_PER_FED + fedCH;
      const auto ipair = cablingMapData.view<SiStripClusterizerConditionsData_fedchSoA>().iPair_(idx);
      const auto ipoff = sistrip::STRIPS_PER_FEDCH * ipair;

      const auto data = chanlocs->input(chan);
      const auto len = chanlocs->length(chan);
      if (extendedPrint) {
        if (chan > 34881) {
          LogDebug("SiStripConditions") << chan << "\t" << fedID << "\t" << (int)fedCH << "\t" << idx << "\t" << ipair
                                        << "\t" << ipoff << "\t"
                                        << "-"
                                        << "\t"
                                        << "-"
                                        << "\t" << len << "\t"
                                        << "-"
                                        << "\n";
        }
      }

      if (data != nullptr && len > 0) {
        auto aoff = chanlocs->offset(chan);
        auto choff = chanlocs->inoff(chan);
        const auto end = choff + len;
        if (extendedPrint) {
          LogDebug("SiStripConditions") << chan << "\t" << fedID << "\t" << (int)fedCH << "\t" << idx << "\t" << ipair
                                        << "\t" << ipoff << "\t" << aoff << "\t" << choff << "\t" << len << "\t"
                                        << ((choff + 1) ^ 7) << "\n";
          LogDebug("SiStripConditions") << "\t\tstripIndex\tgroupLength\taoff\n";
        }
        while (choff < end) {
          auto stripIndex = data[(choff++) ^ 7] + ipoff;
          const auto groupLength = data[(choff++) ^ 7];

          if (extendedPrint)
            LogDebug("SiStripConditions")
                << "\t\t" << (int)stripIndex << "\t\t|" << (int)groupLength << "|\t\t" << aoff << "\t";

          totalentries += (2 + groupLength);
          // initialize as invalid strip
          for (auto i = 0; i < 2; ++i, ++aoff) {
            unpackedStrips_host->stripId(aoff) = 0xFFFF;
            unpackedStrips_host->adc(aoff) = 0;
          }
          for (auto i = 0; i < groupLength; ++i, ++aoff) {
            unpackedStrips_host->stripId(aoff) = stripIndex++;
            unpackedStrips_host->channel(aoff) = chan;
            auto dt = data[(choff++) ^ 7];
            unpackedStrips_host->adc(aoff) = dt;
            if (extendedPrint)
              LogDebug("SiStripConditions")
                  << aoff << ":" << (uint32_t)dt << ":" << (uint32_t)(unpackedStrips_host->adc(aoff)) << " ";
          }
          if (extendedPrint)
            LogDebug("SiStripConditions") << "\n";
        }
      }
    }
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiStripRawToCluster);