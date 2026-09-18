// EDProducer unpacking FEDRawData to create an output collection of Phase2TrackerCluster1D
// for the Phase2 Outer Tracker

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/FEDRawData/interface/SLinkRocketHeaders.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include <unordered_map>

#include "EventFilter/Phase2TrackerRawToDigi/interface/ChannelsMask.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/ChannelsOffset.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/CRACKMapping.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/TrackerBlock.h"

using namespace Phase2TrackerSpecifications;
using namespace Phase2DAQFormatSpecification;


class RawToClusterProducer : public edm::stream::EDProducer<> {
public:
  explicit RawToClusterProducer(const edm::ParameterSet&);
  ~RawToClusterProducer() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  uint32_t get32bWordAtByte(std::span<const unsigned char> data, size_t initByte, size_t startByte, bool debug /* = false */) ;
  TrackerHeader getTrackerHeader(std::span<const unsigned char> data);
  ChannelsMask getChannelMaskingProfile(std::span<const unsigned char> data);
  void dumpPacket(const unsigned char* data, size_t dataSize);
  void readPayload(std::vector<uint32_t>& clusterWords,
                   std::vector<uint32_t>& lines,
                   int numClusters,
                   int& nAvailableBits,
                   int& iLine,
                   int& bitsToRead,
                   int& nFullClusters,
                   int clusterBits,
                   int clusterWordMask,
                   bool isPixelCluster,
                   int nFullClustersStrips = 0);

  int createMask(int nBits);
  std::pair<Phase2TrackerCluster1D, bool> unpack2S(uint32_t, unsigned int);
  Phase2TrackerCluster1D unpackStripOnPS(uint32_t, unsigned int);
  Phase2TrackerCluster1D unpackPixelOnPS(uint32_t, unsigned int);

  void dumpRawFile(const unsigned char*, size_t, bool);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<RawDataBuffer> rawDataBufferToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> OutputClusterCollectionToken_;

  const TrackerDetToDTCELinkCablingMap* cablingMap_ = nullptr;
  const TrackerGeometry* trackerGeometry_ = nullptr;
  const TrackerTopology* trackerTopology_ = nullptr;
  std::map<int, std::pair<int, int>> stackMap_;
  bool analyzeCRACK_;
  std::unique_ptr<crack::CRACKMapping> crackMapping_;
};

RawToClusterProducer::RawToClusterProducer(const edm::ParameterSet& iConfig)
    : rawDataBufferToken_(consumes<RawDataBuffer>(iConfig.getParameter<edm::InputTag>("fedDataBuffer"))),
      cablingMapToken_(
          esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      analyzeCRACK_(iConfig.getParameter<bool>("analyzeCRACK")) {
      
    produces<Phase2TrackerCluster1DCollectionNew>();
    
    if (analyzeCRACK_) {
        crackMapping_ = std::make_unique<crack::CRACKMapping>();
        if (iConfig.exists("crackMapping")) {
            auto mappingVPSet = iConfig.getParameter<std::vector<edm::ParameterSet>>("crackMapping");
            crackMapping_->loadFromVPSet(mappingVPSet);
        } else {
            throw cms::Exception("ConfigurationError") 
                << "analyzeCRACK is true but crackMapping VPSet not found!";
        }
        edm::LogInfo("RawToClusterProducer") << "CRACK mapping loaded successfully";
    }
}

RawToClusterProducer::~RawToClusterProducer() {}

void RawToClusterProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // get cabling from event setup
  cablingMap_ = &iSetup.getData(cablingMapToken_);

  // from the OLD CODE
  // FIXME: build map of stacks to compensate for missing trackertopology methods
  trackerGeometry_ = &iSetup.getData(trackerGeometryToken_);
  trackerTopology_ = &iSetup.getData(trackerTopologyToken_);

  for (auto iu = trackerGeometry_->detUnits().begin(); iu != trackerGeometry_->detUnits().end(); ++iu) {
    unsigned int detId_raw = (*iu)->geographicalId().rawId();
    DetId detId = DetId(detId_raw);
    if (detId.det() == DetId::Detector::Tracker) {
      // build map of upper and lower for each module
      if (trackerTopology_->isLower(detId) != 0) {
        stackMap_[trackerTopology_->stack(detId)].first = detId;
      }
      if (trackerTopology_->isUpper(detId) != 0) {
        stackMap_[trackerTopology_->stack(detId)].second = detId;
      }
    }
  }  // end loop on detunits
}

void RawToClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto outputClusterCollection = std::make_unique<Phase2TrackerCluster1DCollectionNew>();

  const auto& rawDataBuffer = iEvent.get(rawDataBufferToken_);

  auto slink_header_size = sizeof(SLinkRocketHeader_v3);
  auto slink_trailer_size = sizeof(SLinkRocketTrailer_v3);

  TrackerTrailer theTrailer;
  ChannelsOffset theOffsets;

  for (int dtcID = MIN_DTC_ID; dtcID < MAX_DTC_ID + 1; dtcID++) {
    // read the 4 slinks
    for (unsigned int iSlink = 0; iSlink < SLINKS_PER_DTC; iSlink++) {

      unsigned totID = iSlink + SLINKS_PER_DTC * (dtcID - 1) + CMSSW_TRACKER_ID;
      auto const& fedData = rawDataBuffer.fragmentData(totID);
      
      if (fedData.size() > 0 ) {

        auto dataPtr = fedData.payload(slink_header_size, slink_trailer_size);

        TrackerHeader extractedTrackerHeader = getTrackerHeader(dataPtr);
        int coreID = 0;

        // Check if CMSSW can decode the binary.
        if (extractedTrackerHeader.getVersionMajor() == Phase2DAQFormatSpecification::VERSION_MAJOR_V1_0 && 
            extractedTrackerHeader.getVersionMinor() == Phase2DAQFormatSpecification::VERSION_MINOR_V1_0) {
            edm::LogInfo("RawToClusterProducer") << "Read version from binary that is supported. RawToClusterProducer() can decode the binary.";
        } else {
          throw cms::Exception("CMSSW Unpacker RawToClusterProducer() is incopatible with the format version found in this binary. Aborting any further processing.");
        }
        coreID = extractedTrackerHeader.getDAQpathCoreID();

        ChannelsMask ExtractedChannelsMask = getChannelMaskingProfile(dataPtr);

        // read the offsets: each 32 bit word contains two offset words of 16 bit each
        std::vector<uint32_t> offsetWords;
        
        size_t nOffsetsLines = OFFSET_BITS * CICs_PER_SLINK / N_BITS_PER_WORD; 
        size_t initByte = HEADER_N_LINES * N_BYTES_PER_WORD; 
        size_t endByte = (nOffsetsLines - 1) * N_BYTES_PER_WORD + initByte;

        for (size_t i = initByte; i <= endByte; i += N_BYTES_PER_WORD)  { // Read 4 bytes (32 bits) at a time
          uint32_t word32b = get32bWordAtByte(dataPtr, i, initByte, false);
          uint16_t low  = static_cast<uint16_t>(word32b & ((uint32_t{1} << OFFSET_BITS) - 1)); 
          uint16_t high = static_cast<uint16_t>((word32b >> OFFSET_BITS) & ((uint32_t{1} << OFFSET_BITS) - 1));
          offsetWords.push_back(high);
          offsetWords.push_back(low);
        }  
        theOffsets.setValue(offsetWords);
        int initial_offset = initByte + (nOffsetsLines + RESERVED_N_LINES ) * N_BYTES_PER_WORD; // 
        
        uint32_t firstEventID = 0;
        bool firstEventIDSet = false;

        std::vector<Phase2TrackerCluster1D> thisChannel1DSeedClusters, thisChannel1DCorrClusters;
        for (unsigned int iChannel = 0; iChannel < CICs_PER_SLINK; iChannel++) {

          // If this channel is masked, skip it and handle the next.
          if (ExtractedChannelsMask.isChannelMasked(iChannel))
            continue;

          // clear the collection if iChannel is even
          if (iChannel % 2 == 0) {
            thisChannel1DSeedClusters.clear();
            thisChannel1DCorrClusters.clear();
          }

          // retrieve the module type:
          // first we need to construct the DTCElinkId object ## dtc_id, gbtlink_id, elink_id
          // to get the gbt_id we should reverse what is done in the packer function,
          // where clusters from channel X are split into 2*i and 2*i+1 based on being from CIC0 or CIC1
          unsigned int gbt_id = iSlink * MODULES_PER_SLINK + std::div(iChannel, 2).quot;
          if (analyzeCRACK_) {
            gbt_id = crackMapping_->getGbtID(dtcID, coreID, std::div(iChannel, 2).quot);
          }

          DTCELinkId thisDTCElinkId(dtcID, gbt_id, 0);

          int thisDetId = -1;
          bool is2SModule = false;
          // then pass it to the map to get the detid
          if (cablingMap_->knowsDTCELinkId(thisDTCElinkId)) {
            auto possibleDetIds = cablingMap_->dtcELinkIdToDetId(
                thisDTCElinkId);  // this returns a pair, detid will be an uint32_t (not a DetId)
            thisDetId = possibleDetIds->second;
            LogTrace("RawToClusterProducer")
                << "slink: " << iSlink << "\tiDTC: " << unsigned(dtcID) << "\tiGBT: " << unsigned(gbt_id)
                << "\tielink: " << unsigned(0) << "\t -> detId:" << thisDetId;
            // check is 2S or PS
            is2SModule =
                trackerGeometry_->getDetectorType(stackMap_[thisDetId].first) == TrackerGeometry::ModuleType::Ph2SS;
          } else {
            LogTrace("RawToClusterProducer") << "slink: " << iSlink << "\tiDTC: " << unsigned(dtcID)
                                             << "\tiGBT: " << unsigned(gbt_id) << " -> not connected? ";
            continue;
          }

          if (extractedTrackerHeader.is2S() != is2SModule)
            edm::LogError("RawToClusterProducer") << "ERROR: Header for channel " << iChannel << " expects a different type of module";

          // retrieve the channel offset
          int channelOffset = theOffsets.getOffsetForChannel(iChannel);

          size_t idx = initial_offset + channelOffset * N_BYTES_PER_WORD; // (N_BYTES_PER_WORD=4)
          uint32_t headerWord = get32bWordAtByte(dataPtr, idx, initial_offset, false);

          /**
           * HANDLING/CATCHING ERRORS ON DATA READ FROM BINARY
           * @brief: Two checks are currently being perfored:
           * 1) Throw warnings for CIC hard buffer overflows (CIC Event ID = 511). More information in the CIC2.1 Manual:
           * https://edms.cern.ch/ui/#!master/navigator/document?P:100517181:101166963:subDocs, page 30.
           * 2) CIC Event ID should be the same in the entire packet.
           * 3) The user is warned in case a fragment contains at least 1 error in FEs & CIC (Warning).
           */

          /* Check (1) */
          uint32_t eventID = (headerWord >> (N_BITS_PER_WORD - L1ID_BITS)) & L1ID_MAX_VALUE; // 9-bit field
          
          if (eventID == CIC_HARD_BUFFER_OVERFLOW) {
            LogTrace("RawToClusterProducer") << "WARNING: Found CIC Hard Overflow @ Event " << iEvent.id().event() << std::endl;
          }
          
          /* Check (2) */
          if (!firstEventIDSet) {
            firstEventID = eventID;
            firstEventIDSet = true;
          } else if (eventID != firstEventID) {
            throw cms::Exception("CIC Event ID Mismatch! This is a serious issue.") 
                << "CIC Event ID Mismatch Detected! First Event ID: " << firstEventID 
                << ", Current Event ID: " << eventID;
          }

          int channelErrors = (headerWord >> (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS)) & CIC_ERROR_MASK; // 9-bit field
          unsigned int numStripClusters = (headerWord >> (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - N_STRIP_CLUSTER_BITS)) & N_CLUSTER_MASK;
          unsigned int numPixelClusters = (headerWord) & N_CLUSTER_MASK;
          LogTrace("RawToClusterProducer") << "CHANNEL " << iChannel << " HEADER " << std::bitset<32>(headerWord) 
                                           << " (" << channelErrors << " channelErrors, "
                                           << numPixelClusters << " pixel clusters, "
                                           << numStripClusters << " strip clusters)\n";

          /* Check (3) */
          if (channelErrors > 0) {
            LogTrace("RawToClusterProducer") 
                << "WARNING: Channel " << iChannel << " has errors " ;
            continue;
          } else if (is2SModule && numPixelClusters > 0) {
            edm::LogError("RawToClusterProducer") << "ERROR: Header for channel " << iChannel << " expects non-zero ("
                                                  << numPixelClusters << ") pixel clusters on a 2S module\n" <<
                                                  std::bitset<32>(headerWord);
          }

          /************************************************************************************/

          // define the number of lines of the payload
          unsigned int nLines =
              (numStripClusters + numPixelClusters > 0)
                  ? int((numStripClusters * SS_CLUSTER_BITS + numPixelClusters * PX_CLUSTER_BITS) / N_BITS_PER_WORD) + 1
                  : 0;

          if (numStripClusters + numPixelClusters > 0) {
            LogTrace("RawToClusterProducer")
                << "\tchannel " << iChannel << "\theader: " << std::bitset<N_BITS_PER_WORD>(headerWord)
                << "\tn strip clusters = " << numStripClusters << "\tn pixel clusters = " << numPixelClusters
                << " (n lines = " << nLines << ")";
          }

          // first retrieve all lines filled with clusters
          std::vector<uint32_t> lines;
          for (unsigned int iline = 0; iline < nLines; iline++) {
            // header is at channelOffset, so payload starts at channelOffset + 1
            int cluster_payload_idx = channelOffset + 1 + iline;
            size_t bytePos = initial_offset + cluster_payload_idx * N_BYTES_PER_WORD;
            uint32_t word = get32bWordAtByte(dataPtr, bytePos, initial_offset, false);
            lines.push_back(word);
          }
          
          if (lines.size() != nLines) {
            edm::LogError("RawtoClusterProducer")
                << "ERROR: Numbers of stored lines does not match with size of lines to be read!";
            return;
          }

          // first retrieve the cluster words
          // this was uint16, check if can be changed back
          std::vector<uint32_t> stripClustersWords;
          stripClustersWords.resize(numStripClusters);

          std::vector<uint32_t> pixelClustersWords;
          pixelClustersWords.resize(numPixelClusters);

          // create groups of 14 (17) bits for 2S (PS) clusters, joining consecutive lines if needed
          int nAvailableBits = N_BITS_PER_WORD;
          int iLine = 0;
          int bitsToRead = 0;
          int nFullClustersStrip = 0;
          int nFullClustersPix = 0;

          readPayload(stripClustersWords,
                      lines,
                      numStripClusters,
                      nAvailableBits,
                      iLine,
                      bitsToRead,
                      nFullClustersStrip,
                      SS_CLUSTER_BITS,
                      SS_CLUSTER_WORD_MASK,
                      false);
          readPayload(pixelClustersWords,
                      lines,
                      numPixelClusters,
                      nAvailableBits,
                      iLine,
                      bitsToRead,
                      nFullClustersPix,
                      PX_CLUSTER_BITS,
                      PX_CLUSTER_WORD_MASK,
                      true,
                      nFullClustersStrip);

          // unpack the cluster words and create Phase2TrackerCluster1D objects
          int count_clusters = 0;
          if (is2SModule) {
            // create the Phase2TrackerCluster1D objects for 2S modules
            for (auto icluster : stripClustersWords) {
              std::pair<Phase2TrackerCluster1D, bool> thisCluster = unpack2S(icluster, iChannel);
              if (thisCluster.second)
                thisChannel1DSeedClusters.push_back(thisCluster.first);
              else
                thisChannel1DCorrClusters.push_back(thisCluster.first);
              count_clusters++;
            }  // end loop on cluster words
          } else {
            // create the Phase2TrackerCluster1D objects for PS modules
            // first loop on strip clusters
            for (auto icluster : stripClustersWords) {
              Phase2TrackerCluster1D thisCluster = unpackStripOnPS(icluster, iChannel);
              // for PS, strip is always correlated sensor
              thisChannel1DCorrClusters.push_back(thisCluster);
              count_clusters++;
            }
            // then loop on pixel clusters
            for (auto icluster : pixelClustersWords) {
              Phase2TrackerCluster1D thisCluster = unpackPixelOnPS(icluster, iChannel);
              // for PS, pixel is always seed sensor
              thisChannel1DSeedClusters.push_back(thisCluster);
              count_clusters++;
            }
          }

          // use FastFiller to fill the output DetSetVector output collection
          // fill every time that 2 channels are read
          if (iChannel % 2 != 1)
            continue;

          // Store clusters of this channel
          std::vector<Phase2TrackerCluster1D>::iterator it;
          {
            // inner detid is defined as module detid + 1. First int in the pair from the map
            edmNew::DetSetVector<Phase2TrackerCluster1D>::FastFiller spcs(*outputClusterCollection,
                                                                          stackMap_[thisDetId].first);
            for (it = thisChannel1DSeedClusters.begin(); it != thisChannel1DSeedClusters.end(); it++) {
              spcs.push_back(*it);
            }
          }
          {
            // outer detid is defined as inner detid + 1 or module detid + 2. Second int in the pair from the map
            edmNew::DetSetVector<Phase2TrackerCluster1D>::FastFiller spcc(*outputClusterCollection,
                                                                          stackMap_[thisDetId].second);
            for (it = thisChannel1DCorrClusters.begin(); it != thisChannel1DCorrClusters.end(); it++) {
              spcc.push_back(*it);
            }
          }

        }  // end loop on channels for this dtc
        
        // read the tracker trailer
//         std::vector<uint32_t> trailerWords;
//           
//         size_t tracker_trailer_index = dataPtr.size() - TRAILER_N_LINES * N_BYTES_PER_WORD;
//         for (size_t i = tracker_trailer_index; i < tracker_trailer_index + TRAILER_N_LINES * N_BYTES_PER_WORD;
//              i += N_BYTES_PER_WORD)  // Read 4 bytes (32 bits) at a time
//         {
//           // Extract 4 bytes (32 bits) and pack them into a uint32_t word
//           trailerWords.push_back(readLine(dataPtr, i));
//         }
//         theTrailer.setValue(trailerWords);
//         if (theTrailer.is2S() != theHeader.is2S())
//           edm::LogError("RawToClusterProducer") << "ERROR: Header and trailer expect different types of modules";
// 
      }  // end fed data size > 0
    }  // end loop on 4 slink of this dtc
  }  // end loop on dtcs
  LogTrace("RawToClusterProducer") << "output cluster collection contains clusters from "
                                   << outputClusterCollection->size() << " channels";
  iEvent.put(std::move(outputClusterCollection));
}


uint32_t RawToClusterProducer::get32bWordAtByte(std::span<const unsigned char> data,
                                                size_t bytePos,
                                                size_t startByte,
                                                bool debug /* = false */) {
    // word index relative to the beginning of the offset area
    size_t wordIndex = (bytePos - startByte) / N_BYTES_PER_WORD;

    // reverse inside groups of 4 words
    size_t group = wordIndex / 4;
    size_t offset = wordIndex % 4;
    size_t reversedWordIndex = group * 4 + (3 - offset);

    // back to absolute byte position
    size_t byteOffset = startByte + reversedWordIndex * N_BYTES_PER_WORD;

    uint32_t word = (static_cast<uint32_t>(data[byteOffset + 3]) << 24) |
                    (static_cast<uint32_t>(data[byteOffset + 2]) << 16) |
                    (static_cast<uint32_t>(data[byteOffset + 1]) << 8) |
                    (static_cast<uint32_t>(data[byteOffset]));

    if (debug) {
        LogTrace("RawToClusterProducer") << "wordIndex = " << wordIndex 
                                         << "\tbyteOffset = " << byteOffset 
                                         << "\tword = " << std::bitset<32>(word)
                                         << "\t 0x" << std::hex << std::setw(8) << std::setfill('0') << word 
                                         << std::dec;                                         
    }
    return word;
}

/**
 * @brief Returns Tracker Header from the DAQ Packet.
 * @param data Pointer to the raw data buffer (passed by reference)
 * @return TrackerHeader Class Object.
 */
TrackerHeader RawToClusterProducer::getTrackerHeader(std::span<const unsigned char> data) {
    std::vector<uint32_t> words(Phase2DAQFormatSpecification::HEADER_N_LINES);
    size_t startByte = Phase2DAQFormatSpecification::DTC_HEADER_OFFSET * Phase2DAQFormatSpecification::N_BYTES_PER_WORD;
    for (int i = 0; i < Phase2DAQFormatSpecification::HEADER_N_LINES; ++i) {
        words[i] = get32bWordAtByte(data, 
                                    startByte + (i * Phase2DAQFormatSpecification::N_BYTES_PER_WORD), 
                                    startByte, false);
    }
    TrackerHeader captureHeader(words);
    return captureHeader;
}

/**
 * @brief Returns the Channel Masking for this Run
 * @param data Pointer to the raw data buffer (passed by reference)
 * @return ChannelsMask Class Object.
 */
ChannelsMask RawToClusterProducer::getChannelMaskingProfile(std::span<const unsigned char> data) {
    std::array<uint32_t, 2> words;
    for (int i = 0; i < Phase2DAQFormatSpecification::DTC_CHANNEL_MASK_SIZE; ++i) {
      words[i] = get32bWordAtByte(data, 
                                  Phase2DAQFormatSpecification::DTC_CHANNEL_MASK_OFFSET * Phase2DAQFormatSpecification::N_BYTES_PER_WORD + (i * Phase2DAQFormatSpecification::N_BYTES_PER_WORD), 
                                  0, false);
    }
    ChannelsMask captureMasking(words);
    return captureMasking;
}

void RawToClusterProducer::dumpPacket(const unsigned char* data, size_t dataSize) {
    for (size_t l16byteslineID = 0; l16byteslineID < (dataSize + 15) / 16; l16byteslineID++) {
        for (size_t byte_within_line = 0; byte_within_line < 16; byte_within_line++) {
            size_t index = l16byteslineID * 16 + byte_within_line;
            if (index >= dataSize) break;  // Stop if we've printed all bytes
            printf("%02X ", (unsigned int)data[index]);
        }
        printf("\n");
    }
}

std::pair<Phase2TrackerCluster1D, bool> RawToClusterProducer::unpack2S(uint32_t clusterWord, unsigned int iChannel) {
  uint32_t chipID = (clusterWord >> (SS_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;  // 3 bits
  uint32_t sclusterAddress = (clusterWord >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_ONLY_BITS_2S)) &
                             SCLUSTER_ADDRESS_MASK;  // why not uint16?
  bool isSeedSensor =
      (clusterWord >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_2S)) & IS_SEED_SENSOR_MASK;  // 8 bits
  uint32_t width = clusterWord & WIDTH_MAX_VALUE;                                                          // 3 bits
  // cluster width is truncated during packing (3 bits)
  // since width = 0 is unphysical, we can at least recover cluster with width == 8
  // by assuming that clusters packed with width == 0 had in reality width = 8
  // this is a tmp fix, we should maybe think about how to properly do this.
  // also, for original widths > 8: again, due to truncation, they get an incorrect width of
  // cluster.getWidth() & WIDTH_MAX_VALUE. should be probably fixed in the packer
  // (e.g. if width > 8, pack with width = 0)
//   if (width == 0)
//     width = 8;
  LogTrace("RawToClusterProducer") << "\t[unpacking] chipID : " << (chipID) << "\t " << std::bitset<3>(chipID);
  LogTrace("RawToClusterProducer") << "\t[unpacking] address : " << (sclusterAddress) << "\t "
                                   << std::bitset<8>(sclusterAddress);
  LogTrace("RawToClusterProducer") << "\t[unpacking] width : " << (width) << "\t " << std::bitset<3>(width);
  LogTrace("RawToClusterProducer") << "";

  unsigned int x = STRIPS_PER_CBC * chipID + sclusterAddress;
  unsigned int y = iChannel % 2 == 0 ? 0 : 1;

  Phase2TrackerCluster1D thisCluster = Phase2TrackerCluster1D(x, y, width);
  return std::make_pair(thisCluster, isSeedSensor);
}

Phase2TrackerCluster1D RawToClusterProducer::unpackStripOnPS(uint32_t clusterWord, unsigned int iChannel) {
  // FIXME: we don't need uint32 everywere
  uint32_t chipID = (clusterWord >> (SS_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;  // 3 bits
  uint32_t sclusterAddress = (clusterWord >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS)) &
                             SCLUSTER_ADDRESS_PS_MAX_VALUE;  // 7 bits
  uint32_t width = (clusterWord >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS - WIDTH_BITS)) &
                   WIDTH_MAX_VALUE;               // 3 bits
  uint32_t mipBit = clusterWord & MIP_BITS_MASK;  // 1 bits
  // see warning above for how to treat the width
//   if (width == 0)
//     width = 8;
  LogTrace("RawToClusterProducer") << "\t[unpacking] chipID : " << (chipID) << "\t "
                                   << std::bitset<CHIP_ID_BITS>(chipID) << std::endl;
  LogTrace("RawToClusterProducer") << "\t[unpacking] address : " << (sclusterAddress) << "\t "
                                   << std::bitset<SCLUSTER_ADDRESS_BITS_PS>(sclusterAddress) << std::endl;
  LogTrace("RawToClusterProducer") << "\t[unpacking] width : " << (width) << "\t " << std::bitset<WIDTH_BITS>(width)
                                   << std::endl;
  LogTrace("RawToClusterProducer") << "\t[unpacking] mpBit : " << (mipBit) << "\t " << std::bitset<1>(mipBit)
                                   << std::endl;

  unsigned int x = STRIPS_PER_SSA * chipID + sclusterAddress;
  unsigned int y = iChannel % 2 == 0 ? 0 : 1;

  return Phase2TrackerCluster1D(x, y, width, mipBit);
}

Phase2TrackerCluster1D RawToClusterProducer::unpackPixelOnPS(uint32_t clusterWord, unsigned int iChannel) {
  // FIXME: we don't need uint32 everywere
  uint32_t chipID = (clusterWord >> (PX_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;  // 3 bits
  uint32_t sclusterAddress = (clusterWord >> (PX_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS)) &
                             SCLUSTER_ADDRESS_PS_MAX_VALUE;  // why not uint16?
  uint32_t width = (clusterWord >> (PX_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS - WIDTH_BITS)) &
                   WIDTH_MAX_VALUE;  // 3 bits
//   see warning above for how to treat the width
//   if (width == 0)
//     width = 8;
  uint32_t z = clusterWord & PS_Z_BITS_MASK;  // 4 bits

  LogTrace("RawToClusterProducer") << "\t[unpacking] chipID : " << (chipID) << "\t "
                                   << std::bitset<CHIP_ID_BITS>(chipID);
  LogTrace("RawToClusterProducer") << "\t[unpacking] address : " << (sclusterAddress) << "\t "
                                   << std::bitset<SCLUSTER_ADDRESS_BITS_PS>(sclusterAddress);
  LogTrace("RawToClusterProducer") << "\t[unpacking] width : " << (width) << "\t " << std::bitset<WIDTH_BITS>(width);
  LogTrace("RawToClusterProducer") << "\t[unpacking] z : " << (z) << "\t " << std::bitset<4>(z);

  unsigned int x = STRIPS_PER_SSA * chipID + sclusterAddress;
  // NB: not really clear from the packer code. Got it from the old code.
  unsigned int y = iChannel % 2 == 0 ? z : (z + 16);
  // (chipId() >= MAX_CBC_PER_FE / 2) ? (rawY() + PS_COLS / 2) : rawY();

  return Phase2TrackerCluster1D(x, y, width);
}

// create groups of 14/17 bits, joining consecutive lines if needed
// each 2S cluster payload consists of 3bits for chipID, 8 bits for address, 3 bits for width = 14 bits
// each P on PS cluster payload consists of 3bits for chipID, 7 bits for address, 1 bit for mpBit, 3 bits for width = 17 bits
// each S on PS cluster payload consists of 3bits for chipID, 8 bits for address, 1 bit for z, 3 bits for width = 17 bits
void RawToClusterProducer::readPayload(std::vector<uint32_t>& clusterWords,
                                       std::vector<uint32_t>& lines,
                                       int numClusters,
                                       int& nAvailableBits,
                                       int& iLine,
                                       int& bitsToRead,
                                       int& nFullClusters,
                                       int clusterBits,
                                       int clusterWordMask,
                                       bool isPixelCluster,
                                       int nFullClustersStrips) {
  for (int icluster = 0; icluster < numClusters; icluster++) {
    if (nAvailableBits >= clusterBits) {
      // calculate the shift
      int shift = N_BITS_PER_WORD - bitsToRead - (nFullClusters + 1) * clusterBits;
      // take into account bits already used for the last strip cluster
      if (icluster == 0 && isPixelCluster)
        shift -= (nFullClustersStrips)*SS_CLUSTER_BITS;
      nFullClustersStrips = 0;  // reset

      // mask, and save cluster word
      clusterWords[icluster] = (lines[iLine] >> shift) & clusterWordMask;
      // update available bits and number of full clusters from this line
      nAvailableBits -= clusterBits;
      nFullClusters++;

      if (nAvailableBits == 0) {
        iLine++;
        nAvailableBits = N_BITS_PER_WORD;
        nFullClusters = 0;
        bitsToRead = 0;
      }
    } else {
      // get the remaining bits from the current line. first create the mask, then mask
      int nMask = createMask(nAvailableBits);
      uint16_t wordLeft = lines[iLine] & nMask;

      // create mask for next line
      bitsToRead = clusterBits - nAvailableBits;
      int nextMask = createMask(bitsToRead);
      // shift and mask
      uint16_t wordRight = (lines[iLine + 1] >> (N_BITS_PER_WORD - bitsToRead)) & nextMask;

      // compose the full cluster word
      clusterWords[icluster] = (wordLeft << bitsToRead) | wordRight;

      // re-set n available bits
      nAvailableBits = N_BITS_PER_WORD - bitsToRead;
      // advance by one line and re-init the number of complete clusters read from the current line
      iLine++;
      nFullClusters = 0;
    }
  }
}


void RawToClusterProducer::dumpRawFile(const unsigned char* dataPtr, size_t data_size, bool hexa) {
    
  if (hexa) {
    for (size_t i = 0; i < data_size; i += 16) {
      std::ostringstream line;
      line << std::hex
           << std::setw(7)
           << std::setfill('0')
           << std::nouppercase
           << i
           << " ";

      for (size_t j = i; j < i + 16 && j < data_size; j += 2)
      {
        uint16_t word = static_cast<uint16_t>(static_cast<uint8_t>(dataPtr[j])) << 8;
        if (j + 1 < data_size)
          word |= static_cast<uint8_t>(dataPtr[j + 1]);

        line << std::hex
             << std::setw(4)
             << std::setfill('0')
             << std::nouppercase
             << word;

        if (j + 2 < i + 16 && j + 2 < data_size)
          line << " ";
      }
      LogTrace("RawToClusterProducer") << line.str();
    }
  } else {
    for (size_t i = 0; i < data_size; i += 8) {
      std::ostringstream line;

      line << "  ";
      for (size_t j = i; j < i + 8 && j < data_size; ++j) {
        std::bitset<8> bits(dataPtr[j]);
        line << "  " << bits << " ";
      }
      line << std::dec << std::min(i + 8, data_size) << "  ";
      LogTrace("RawToClusterProducer") << line.str();
    }
  }
}


int RawToClusterProducer::createMask(int nBits) { return (1 << nBits) - 1; }

DEFINE_FWK_MODULE(RawToClusterProducer);
