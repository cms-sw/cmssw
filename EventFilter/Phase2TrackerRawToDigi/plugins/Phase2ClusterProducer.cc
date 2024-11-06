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

// temporary definition of other classes
class TrackerHeader {
public:
    std::vector<uint32_t> values_{std::vector<uint32_t>(4,0)};

    void setValue(std::vector<uint32_t>& newValues) {
      values_ = newValues;
    }

    void printValues() const {
      for (size_t i = 0; i < values_.size(); ++i) {
        std::cout << "TrackerHeader[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) <<  std::endl;
      }
    }   
    void printValue(size_t i) const {
      std::cout << "TrackerHeader[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) <<  std::endl;
    }   
};

class ChannelsOffset {
public:
    std::vector<uint32_t> values_;
    std::vector<uint16_t> offsetMap_{std::vector<uint16_t>(36,0)};

    void setValue(std::vector<uint32_t>& newValues) {
      values_ = newValues;
      fillOffsetMap();
    }
    
    void printValues() const {
      for (size_t i = 0; i < values_.size(); ++i) {
        std::cout << "ChannelsOffset[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) <<  std::endl;
      }
    }   
    void printValue(size_t i) const {
      std::cout << "ChannelsOffset[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) <<  std::endl;
    }   
    
    void fillOffsetMap(){
      for (size_t i = 0; i < 18; ++i) {
// //       for (size_t i = 0; i < values_.size(); ++i) {
        // extract the lower 16 bits by masking with 0xFFFF
       offsetMap_[i*2] = static_cast<uint16_t>(values_[i] & 0xFFFF);
        // extract the upper 16 bits by shifting right by 16
        offsetMap_[i*2+1] =  static_cast<uint16_t>(values_[i] >> 16) ; 
      }
    }

    uint16_t getOffsetForChannel(unsigned int iChannel){
      if (iChannel > 35) {
        throw cms::Exception("Phase2TClusterProducer") << " iChannel " << iChannel << " too high";
      }
      return offsetMap_[iChannel];
    }

    void printMap() const {
      for (size_t i = 0; i < offsetMap_.size(); ++i) {
        std::cout << "offsetMap[" << i << "]: " << offsetMap_[i] << std::endl;
     }
    }   
};

class Phase2ClusterProducer : public edm::stream::EDProducer<> {
public:
    explicit Phase2ClusterProducer(const edm::ParameterSet&);
    ~Phase2ClusterProducer() override;
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    
    int getLineIndex(int channelIdx, unsigned int iline);


private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    
    const edm::EDGetTokenT<FEDRawDataCollection> fedRawDataToken_;
    const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;
    edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> OutputClusterCollectionToken_;
    
    const int nBytes = 4;

    const TrackerDetToDTCELinkCablingMap* cablingMap_ = nullptr;
    const TrackerGeometry* trackerGeometry_ = nullptr;
    const TrackerTopology* trackerTopology_ = nullptr;
    std::map<int, std::pair<int, int>> stackMap_;

};



Phase2ClusterProducer::Phase2ClusterProducer(const edm::ParameterSet& iConfig) : 
      fedRawDataToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("fedRawDataCollection"))),
      cablingMapToken_(esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>())
{
    produces<Phase2TrackerCluster1DCollectionNew>();
}

Phase2ClusterProducer::~Phase2ClusterProducer() 
{

}

void Phase2ClusterProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
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

void Phase2ClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
    auto outputClusterCollection = std::make_unique<Phase2TrackerCluster1DCollectionNew>();

    edm::Handle<FEDRawDataCollection> fedRawDataCollection;
    iEvent.getByToken(fedRawDataToken_, fedRawDataCollection);
  
    TrackerHeader theHeader;
    ChannelsOffset theOffsets;

    int lineLength = 32;
    int clusterWordLength = 14;

    if (!fedRawDataCollection.isValid()) 
    {
      edm::LogError("Phase2DAQAnalyzer") << "No FEDRawDataCollection found!";
      return;
    }

    // Read one entire DTC (#dtcID), as per the producer logic
    unsigned int dtcID = 6;  
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
        for (size_t i = 0; i < 16; i += nBytes)  // Read 4 bytes (32 bits) at a time
        {
          // Extract 4 bytes (32 bits) and pack them into a uint32_t word
          uint32_t word = (static_cast<uint32_t>(dataPtr[i]) << 24) | 
                          (static_cast<uint32_t>(dataPtr[i + 1]) << 16) | 
                          (static_cast<uint32_t>(dataPtr[i + 2]) << 8) | 
                          (static_cast<uint32_t>(dataPtr[i + 3]));
                            
          headerWords.push_back(word);
        }
        theHeader.setValue(headerWords);
        theHeader.printValues() ;
  
        // read the offsets 
        // they start from the fifth line (4 * (5-1))
        // until line 22nd (4 * 22-1)
        std::vector<uint32_t> offsetWords;
        for (size_t i = 16; i < 84; i += nBytes)  // Read 4 bytes (32 bits) at a time
        {
            // Extract 4 bytes (32 bits) and pack them into a uint32_t word
          uint32_t word = (static_cast<uint32_t>(dataPtr[i]) << 24) | 
                          (static_cast<uint32_t>(dataPtr[i + 1]) << 16) | 
                          (static_cast<uint32_t>(dataPtr[i + 2]) << 8) | 
                          (static_cast<uint32_t>(dataPtr[i + 3]));
          offsetWords.push_back(word)      ;          
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
          int idx = 88 + theOffsets.getOffsetForChannel(iChannel) * nBytes;
          // get the channel header
          uint32_t headerWord = (static_cast<uint32_t>(dataPtr[idx]) << 24) | 
                                (static_cast<uint32_t>(dataPtr[idx + 1]) << 16) | 
                                (static_cast<uint32_t>(dataPtr[idx + 2]) << 8) | 
                                (static_cast<uint32_t>(dataPtr[idx + 3]));
          
          unsigned long eventID = (headerWord >> 23) & 0x1FF;       // eventID is a 9-bit field
          int channelErrors = (headerWord >> 14) & 0x1FF;           // channelErrors is a 9-bit field
          unsigned int numClusters = (headerWord >> 7) & 0x7F;      // numClusters is a 7-bit field
          // we'll have separated num strip and num pixel clusters for PS modules, when ready
          // other 7 bits are missing for this line
          
          unsigned int nLines = numClusters > 0 ? int(numClusters * clusterWordLength / lineLength) + 1 : 0;
          if (numClusters > 0){
            std::cout << "channel " << iChannel << "\t header: " << std::bitset<32>(headerWord) <<  std::endl;
            std::cout << "\t n clusters = " << numClusters ;
            std::cout << " -> n lines = " << nLines <<  std::endl;
          }  
          
          std::vector<uint16_t> clustersWords;
          clustersWords.resize(numClusters);
          std::vector<Phase2TrackerCluster1D> thisChannel1DClusters;
  
          // first retrieve all lines filled with clusters
          std::vector<uint32_t> lines;
          for (unsigned int iline = 0; iline < nLines; iline++)
          {
            int lineIdx = getLineIndex(idx, iline);
            uint32_t lineWord = (static_cast<uint32_t>(dataPtr[lineIdx]) << 24) | 
                                (static_cast<uint32_t>(dataPtr[lineIdx + 1]) << 16) | 
                                (static_cast<uint32_t>(dataPtr[lineIdx + 2]) << 8) | 
                                (static_cast<uint32_t>(dataPtr[lineIdx + 3]));
            lines.push_back(lineWord);
          }        
          if ( lines.size() != nLines) 
            std::cout << "warning, something went wrong when storing lines " << std::endl;
  
  
          // then create groups of 14 bits, joining consecutive lines if needed
          // each cluster payload consists in 3bits for chipID, 8 bits for address, 3 bits for width = 14 bits
          int nAvailableBits = lineLength;
          int iline = 0;
          int bitsToRead = 0;
          int nFullClusters = 0;
          for (unsigned int icluster = 0; icluster < numClusters; icluster++)
          {
            if (nAvailableBits >= clusterWordLength)
            {
              // calculate the shift 
              int shift = lineLength - bitsToRead - (nFullClusters + 1) * clusterWordLength;
              std::cout << "\t\t cluster " << icluster  << "  shift by " << shift;
//               uint16_t cluster_tmp = (lines[iline] >> shift) & 0x3FFF;
              clustersWords[icluster] = (lines[iline] >> shift) & 0x3FFF;
              std::cout << "\t clusterword " << std::bitset<14>(clustersWords[icluster]) <<  std::endl;
              // and update available bits and number of full clusters from this line
              nAvailableBits -= clusterWordLength;
              nFullClusters++;
//               std::cout << "\t\t\t remaining bits " << nAvailableBits <<  std::endl;
              if (nAvailableBits == 0) {
                  iline++;
                  nAvailableBits = lineLength;
                  nFullClusters = 0;
                  bitsToRead = 0;
              }
            } else {
              // get the remaining bits from this line. first create the mask, then mask 
              int nMask = 0;
              for (int i = 0; i < nAvailableBits; i ++) { 
                nMask |= (1 << i);
              }          
              uint16_t wordLeft = (lines[iline]) & nMask; 
  
              // create mask for next line
              int nMask_newLine = 0;
              bitsToRead = clusterWordLength - nAvailableBits;
              for (int i = 0; i < bitsToRead; i ++) { 
                nMask_newLine |= (1 << i);
              }
              // shift and mask
              uint16_t wordRight = (lines[iline+1] >> (lineLength - bitsToRead)) & nMask_newLine; 
              // compose the full cluster word
              clustersWords[icluster] = ((wordLeft << bitsToRead) | wordRight);  
              std::cout << "\t\t cluster " << icluster << "  on 2 lines \t clusterword " << std::bitset<14>(clustersWords[icluster]) <<  std::endl;
              
              // reset n available bits
              nAvailableBits = lineLength - bitsToRead ;
//               std::cout << "\t\t\t remaining bits " << nAvailableBits <<  std::endl;
              // advance by one line and re-init the number of complete clusters read from this line
              iline++;
              nFullClusters = 0;
            }
          } // end loop storing cluster words
          if ( clustersWords.size() != numClusters) {
            std::cout << "warning, something went wrong when storing clusters " << std::endl;
          }  

          // now create the Phase2TrackerCluster1D object 
          int count_clusters = 0;
          for (auto icluster : clustersWords){
            unsigned int chipID = (unsigned int)((icluster >> 11) & 0x7);
            unsigned int sclusterAddress = (unsigned int)((icluster >> 3) & 0xFF);
            unsigned int width = (unsigned int)((icluster &  0x7));
            // FIXME: currently using random y value of 1
            // 127 is valid only for 2S modules, that is what we have for now
            // for the position, probably need different unpacking logic for 2S and PS
            unsigned int x = 127 * sclusterAddress + chipID;
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
          DTCELinkId thisDTCElinkId(dtcID, iChannel, 0);
          // then pass it to the map to get the detid
          std::cout << "\tiDTC: " << unsigned(dtcID) << " \tiGBT:  " <<  unsigned(iChannel) << " \tielink: " <<  unsigned(0);
          if (cablingMap_->knowsDTCELinkId(thisDTCElinkId))
          {
            auto possibleDetIds = cablingMap_->dtcELinkIdToDetId(thisDTCElinkId);
            std::cout << "\t -> detId:" <<  possibleDetIds->second << std::endl;
            // Store clusters of this channel.
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

int Phase2ClusterProducer::getLineIndex(int channelIdx, unsigned int iline){
  return channelIdx + nBytes + iline * nBytes; 
}

DEFINE_FWK_MODULE(Phase2ClusterProducer);