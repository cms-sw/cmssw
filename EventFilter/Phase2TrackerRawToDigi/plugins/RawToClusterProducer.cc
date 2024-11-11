#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include <unordered_map>

#include "EventFilter/Phase2TrackerRawToDigi/interface/Cluster.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/TrackerHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/ChannelsOffset.h"

// namespace to be added

constexpr uint16_t N_BYTES = 4;
// constexpr int LINE_LENGTH = 32;
constexpr int CLUSTER_LENGTH = 14;
constexpr int N_STRIPS_CBC = 127;


class RawToClusterProducer : public edm::stream::EDProducer<> {
public:
    explicit RawToClusterProducer(const edm::ParameterSet&);
    ~RawToClusterProducer() override;
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    
    int getLineIndex(int channelIdx, unsigned int iline);
    uint32_t readLine(const unsigned char* dataPtr, int lineIdx);


private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    
    const edm::EDGetTokenT<FEDRawDataCollection> fedRawDataToken_;
    const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
    edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> OutputClusterCollectionToken_;
    
    const TrackerDetToDTCELinkCablingMap* cablingMap_ = nullptr;
    const TrackerGeometry* trackerGeometry_ = nullptr;
    const TrackerTopology* trackerTopology_ = nullptr;
    std::map<int, std::pair<int, int>> stackMap_;

};



RawToClusterProducer::RawToClusterProducer(const edm::ParameterSet& iConfig) : 
      fedRawDataToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("fedRawDataCollection"))),
      cablingMapToken_(esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>())
{
    produces<Phase2TrackerCluster1DCollectionNew>();
}

RawToClusterProducer::~RawToClusterProducer() 
{

}

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

void RawToClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
    auto outputClusterCollection = std::make_unique<Phase2TrackerCluster1DCollectionNew>();

    edm::Handle<FEDRawDataCollection> fedRawDataCollection;
    iEvent.getByToken(fedRawDataToken_, fedRawDataCollection);
  
    TrackerHeader theHeader;
    ChannelsOffset theOffsets;


    if (!fedRawDataCollection.isValid()) 
    {
      edm::LogError("Phase2DAQAnalyzer") << "No FEDRawDataCollection found!";
      return;
    }

    // Read one entire DTC (#dtcID), as per the producer logic
    unsigned int dtcID = 180;  
    // read the 4 slinks
    for (unsigned int iSlink = 0; iSlink < 4; iSlink++)
    {
      // as defined in the DAQProducer code
      unsigned totID = iSlink + 4 * (dtcID - 1) + 0 ;
      const FEDRawData& fedData = fedRawDataCollection->FEDData(totID);  // FED ID 0
      if (fedData.size() > 0) 
      {
        const unsigned char* dataPtr = fedData.data();
        
        // read the header  
        std::vector<uint32_t> headerWords;
        for (size_t i = 0; i < 16; i += N_BYTES)  // Read 4 bytes (32 bits) at a time
        {
          // Extract 4 bytes (32 bits) and pack them into a uint32_t word
          headerWords.push_back(readLine(dataPtr, i));
        }
        theHeader.setValue(headerWords);
//         theHeader.printValues() ;
  
        // read the offsets 
        // they start from the fifth line (4 * (5-1))
        // until line 22nd (4 * 22-1)
        std::vector<uint32_t> offsetWords;
        for (size_t i = 16; i < 84; i += N_BYTES)  // Read 4 bytes (32 bits) at a time
        {
          // Extract 4 bytes (32 bits) and pack them into a uint32_t word
          offsetWords.push_back(readLine(dataPtr, i));          
        }
        theOffsets.setValue(offsetWords);
  //       theOffsets.printMap();
        // then there are 2 reserved lines
        
        // tmp to check the map
  //       std::vector<DTCELinkId> known_dets = cablingMap_->getKnownDTCELinkIds();
  //       for (auto idet : known_dets){
  //         std::cout << "iDTC: " << unsigned(idet.dtc_id()) << " \t iGBT:  " <<  unsigned(idet.gbtlink_id()) << " \t  ielink: " <<  unsigned(idet.elink_id()) << std::endl;
  //       }
        
        // now read the payload (channel header + clusters)
        // assuming all channel headers are there, even if 0 clusters are found
        for (unsigned int iChannel = 0; iChannel < 36; iChannel++)
        {
          // find the channel offset
          // theOffsets.printValue(iChannel);
          int idx = 88 + theOffsets.getOffsetForChannel(iChannel) * N_BYTES;
          // get the channel header
          uint32_t headerWord = readLine(dataPtr, idx);
          
          unsigned long eventID = (headerWord >> 23) & 0x1FF;       // eventID is a 9-bit field
          int channelErrors = (headerWord >> 14) & 0x1FF;           // channelErrors is a 9-bit field
          unsigned int numClusters = (headerWord >> 7) & 0x7F;      // numClusters is a 7-bit field
          // we'll have separate num strip and num pixel clusters for PS modules, when ready
          // other 7 bits are missing for this line
          
          unsigned int nLines = numClusters > 0 ? int(numClusters * CLUSTER_LENGTH / LINE_LENGTH) + 1 : 0;
          if (numClusters > 0){
            std::cout << "channel " << iChannel << "\t header: " << std::bitset<32>(headerWord) <<  std::endl;
//             std::cout << "\t n clusters = " << numClusters ;
//             std::cout << " -> n lines = " << nLines <<  std::endl;
          }  
          
          std::vector<uint16_t> clustersWords;
          clustersWords.resize(numClusters);
          std::vector<Phase2TrackerCluster1D> thisChannel1DClusters;
  
          // first retrieve all lines filled with clusters
          std::vector<uint32_t> lines;
          for (unsigned int iline = 0; iline < nLines; iline++)
          {
            int lineIdx = getLineIndex(idx, iline);
            lines.push_back(readLine(dataPtr, lineIdx));
          }        
          if ( lines.size() != nLines) 
            std::cout << "warning, something went wrong when storing lines " << std::endl;
  
  
          // then create groups of 14 bits, joining consecutive lines if needed
          // each cluster payload consists in 3bits for chipID, 8 bits for address, 3 bits for width = 14 bits
          int nAvailableBits = LINE_LENGTH;
          int iLine = 0;
          int bitsToRead = 0;
          int nFullClusters = 0;
          for (unsigned int icluster = 0; icluster < numClusters; icluster++)
          {
            if (nAvailableBits >= CLUSTER_LENGTH)
            {
              // calculate the shift 
              int shift = LINE_LENGTH - bitsToRead - (nFullClusters + 1) * CLUSTER_LENGTH;
//               std::cout << "\t\t cluster " << icluster  << "  shift by " << shift;
              clustersWords[icluster] = (lines[iLine] >> shift) & 0x3FFF;
//               std::cout << "\t clusterword " << std::bitset<CLUSTER_LENGTH>(clustersWords[icluster]) <<  std::endl;
              // and update available bits and number of full clusters from this line
              nAvailableBits -= CLUSTER_LENGTH;
              nFullClusters++;
//               std::cout << "\t\t\t remaining bits " << nAvailableBits <<  std::endl;
              if (nAvailableBits == 0) {
                  iLine++;
                  nAvailableBits = LINE_LENGTH;
                  nFullClusters = 0;
                  bitsToRead = 0;
              }
            } else {
              // get the remaining bits from this line. first create the mask, then mask 
              int nMask = 0;
              for (int i = 0; i < nAvailableBits; i ++) { 
                nMask |= (1 << i);
              }          
              uint16_t wordLeft = (lines[iLine]) & nMask; 
  
              // create mask for next line
              int nMask_newLine = 0;
              bitsToRead = CLUSTER_LENGTH - nAvailableBits;
              for (int i = 0; i < bitsToRead; i ++) { 
                nMask_newLine |= (1 << i);
              }
              // shift and mask
              uint16_t wordRight = (lines[iLine+1] >> (LINE_LENGTH - bitsToRead)) & nMask_newLine; 
              // compose the full cluster word
              clustersWords[icluster] = ((wordLeft << bitsToRead) | wordRight);  
//               std::cout << "\t\t cluster " << icluster << "  on 2 lines \t clusterword " << std::bitset<CLUSTER_LENGTH>(clustersWords[icluster]) <<  std::endl;
              
              // reset n available bits
              nAvailableBits = LINE_LENGTH - bitsToRead ;
//               std::cout << "\t\t\t remaining bits " << nAvailableBits <<  std::endl;
              // advance by one line and re-init the number of complete clusters read from this line
              iLine++;
              nFullClusters = 0;
            }
          } // end loop storing cluster words
          if ( clustersWords.size() != numClusters) {
            std::cout << "warning, something went wrong when storing clusters " << std::endl;
          }  

          // now create the Phase2TrackerCluster1D object 
          int count_clusters = 0;
          for (auto icluster : clustersWords){
            uint32_t chipID = (icluster >> 11) & 0x7;
            uint32_t sclusterAddress = (icluster >> 3) & 0xFF;
            uint32_t width = icluster &  0x7;

//             std::cout << "[unpacking] chipID : " <<  (chipID) << "\t " << std::bitset<3>(chipID) <<   std::endl;
//             std::cout << "[unpacking] address : " << (sclusterAddress) << "\t " << std::bitset<8>(sclusterAddress) <<   std::endl;
//             std::cout << "[unpacking] width : " << (width)   << "\t " << std::bitset<3>(width) <<   std::endl;
//             std::cout <<  std::endl;
            // FIXME: currently using random y value of 1
            // 127 is valid only for 2S modules, that is what we have for now
            // for the position, probably need different unpacking logic for 2S and PS
            unsigned int x = N_STRIPS_CBC * sclusterAddress + chipID;
            unsigned int y = iChannel%2 == 0 ? 0 : 1;
            std::cout << "\t\t cluster#" << count_clusters << "\t x y width : " << x << " " << y << " " << width;

            Phase2TrackerCluster1D thisCluster = Phase2TrackerCluster1D(x, y, width);
            std::cout << "\t\t check: width from Phase2TrackerCluster1D: " << thisCluster.size() << std::endl;
            thisChannel1DClusters.push_back(thisCluster);
            count_clusters++;
            
          } // end loop on cluster words 
          
          if (count_clusters == 0 ) continue;
  
          // put the filler here, for the output collection
          // need the detid of this module
          // first I need to construct the DTCElinkId object ## dtc_id, gbtlink_id, elink_id
          // to get the gbt_id I should reverse what is done in the DTCUnit.convertToRawData function,
          // where clusters from channel X are split into 2*i and 2*i+1 based on being from CIC0 or CIC1
          unsigned int gbt_id = iSlink * 18 + std::div(iChannel, 2).quot;
          DTCELinkId thisDTCElinkId(dtcID, gbt_id, 0);
          // then pass it to the map to get the detid
          std::cout << "\tslink: " << iSlink <<  "\tiDTC: " << unsigned(dtcID) << " \tiGBT:  " <<  unsigned(gbt_id) << " \tielink: " <<  unsigned(0);
          if (cablingMap_->knowsDTCELinkId(thisDTCElinkId))
          {
            auto possibleDetIds = cablingMap_->dtcELinkIdToDetId(thisDTCElinkId); // this returns a pair, detid will be an uint32_t (not a DetId)
            std::cout << "\t -> detId:" <<  possibleDetIds->second << std::endl;
            // Store clusters of this channel
            // FIXME: we should split them by top and bottom sensors (to detIDs)
            std::vector<Phase2TrackerCluster1D>::iterator it;
            {
              // outer detid is defined as inner detid + 1 or module detid + 2
              edmNew::DetSetVector<Phase2TrackerCluster1D>::FastFiller spct(*outputClusterCollection, stackMap_[possibleDetIds->second].second);
              for (it = thisChannel1DClusters.begin(); it != thisChannel1DClusters.end(); it++) {
                spct.push_back(*it);
              }
            }
          }// if detId is found 
  
        } // end loop on channels for this dtc
      } // end fed data size > 0
    } // end loop on 4 slink of this dtc  
    std::cout << "output cluster collection contains clusters from " << outputClusterCollection->size() << " channels" <<  std::endl;
    iEvent.put(std::move(outputClusterCollection));

}

int RawToClusterProducer::getLineIndex(int channelIdx, unsigned int iline){
    return channelIdx + N_BYTES + iline * N_BYTES; 
}

uint32_t RawToClusterProducer::readLine(const unsigned char* dataPtr, int lineIdx){
    uint32_t line = (static_cast<uint32_t>(dataPtr[lineIdx]) << 24) | 
                    (static_cast<uint32_t>(dataPtr[lineIdx + 1]) << 16) | 
                    (static_cast<uint32_t>(dataPtr[lineIdx + 2]) << 8) | 
                    (static_cast<uint32_t>(dataPtr[lineIdx + 3]));

    return line;                                
}                                


DEFINE_FWK_MODULE(RawToClusterProducer);