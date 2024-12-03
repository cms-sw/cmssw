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

#include "EventFilter/Phase2TrackerRawToDigi/interface/TrackerHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/ChannelsOffset.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"

using namespace Phase2TrackerSpecifications;
using namespace Phase2DAQFormatSpecification;

// namespace to be added


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
    unsigned int dtcID = 180; // 180 for single mu displ
    // read the 4 slinks
    for (unsigned int iSlink = 0; iSlink < SLINKS_PER_DTC; iSlink++)
    {
      // as defined in the DAQProducer code
      unsigned totID = iSlink + SLINKS_PER_DTC * (dtcID - 1) + 0 ;
      const FEDRawData& fedData = fedRawDataCollection->FEDData(totID);  
      if (fedData.size() > 0) 
      {
        const unsigned char* dataPtr = fedData.data();
        
        // read the header  
        std::vector<uint32_t> headerWords;
        for (size_t i = 0; i < HEADER_N_LINES*N_BYTES_PER_WORD; i += N_BYTES_PER_WORD)  // Read 4 bytes (32 bits) at a time
        {
          // Extract 4 bytes (32 bits) and pack them into a uint32_t word
          headerWords.push_back(readLine(dataPtr, i));
        }
        theHeader.setValue(headerWords);
//         theHeader.printValues() ;
  
        // read the offsets 
        // they start from the fifth line (4 * (5-1)) until line 22nd (4 * 22-1)
        std::vector<uint32_t> offsetWords;
        size_t nOffsetsLines = OFFSET_BITS * CICs_PER_SLINK / N_BITS_PER_WORD;
        size_t initByte = HEADER_N_LINES*N_BYTES_PER_WORD;
        size_t endByte = (nOffsetsLines-1)*N_BYTES_PER_WORD + initByte;  // -1 because we only need the starting i of the line

        for (size_t i = initByte; i <= endByte; i += N_BYTES_PER_WORD)  // Read 4 bytes (32 bits) at a time
          offsetWords.push_back(readLine(dataPtr, i));          
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
        // this is not really the channel number, as in the rawToData conversion is channel is split in CIC0_CIC1
        // NOTE: I need to save into the Phase2TrackerCluster1D collection two "channels" at the time 
        // in order to get all the clusters from the same lpGBT and fill them once at the end
        std::vector<Phase2TrackerCluster1D> thisChannel1DSeedClusters, thisChannel1DCorrClusters;
        for (unsigned int iChannel = 0; iChannel < CICs_PER_SLINK; iChannel++)
        {
          // clear the collection if iChannel is even
          if (iChannel%2==0){
            thisChannel1DSeedClusters.clear();
            thisChannel1DCorrClusters.clear();
          }  
          // find the channel offset
          // theOffsets.printValue(iChannel);
          int idx = 88 + theOffsets.getOffsetForChannel(iChannel) * N_BYTES_PER_WORD;
          // get the channel header
          uint32_t headerWord = readLine(dataPtr, idx);
          
          unsigned long eventID = (headerWord >> (N_BITS_PER_WORD - L1ID_BITS)) & L1ID_MAX_VALUE;       // eventID is a 9-bit field
          int channelErrors = (headerWord >> (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS)) & 0x1FF;   // channelErrors is a 9-bit field
          unsigned int numStripClusters = (headerWord >> (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - N_STRIP_CLUSTER_BITS)) & 0x7F;      // numStripClusters is a 7-bit field
          unsigned int numPixelClusters = (headerWord) & 0x7F;      // numStripClusters is a 7-bit field
          
          unsigned int nLines = (numStripClusters + numPixelClusters > 0) ? 
                                 int((numStripClusters * SS_CLUSTER_BITS + numPixelClusters * PX_CLUSTER_BITS)/ N_BITS_PER_WORD) + 1 : 0;

          if (numStripClusters + numPixelClusters > 0 ){
            std::cout << "channel " << iChannel << "\t header: " << std::bitset<N_BITS_PER_WORD>(headerWord) <<  std::endl;
            std::cout << "\t n strip clusters = " << numStripClusters ;
            std::cout << "\t n pixel clusters = " << numPixelClusters ;
            std::cout << " (n lines = " << nLines << ")" <<  std::endl;
          }  
          
          std::vector<uint16_t> stripClustersWords;
          stripClustersWords.resize(numStripClusters);
  
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
          int nAvailableBits = N_BITS_PER_WORD;
          int iLine = 0;
          int bitsToRead = 0;
          int nFullClusters = 0;
          for (unsigned int icluster = 0; icluster < numStripClusters; icluster++)
          {
            if (nAvailableBits >= SS_CLUSTER_BITS)
            {
              // calculate the shift 
              int shift = N_BITS_PER_WORD - bitsToRead - (nFullClusters + 1) * SS_CLUSTER_BITS;
//               std::cout << "\t\t cluster " << icluster  << "  shift by " << shift;
              stripClustersWords[icluster] = (lines[iLine] >> shift) & 0x3FFF;
//               std::cout << "\t clusterword " << std::bitset<SS_CLUSTER_BITS>(stripClustersWords[icluster]) <<  std::endl;
              // and update available bits and number of full clusters from this line
              nAvailableBits -= SS_CLUSTER_BITS;
              nFullClusters++;
//               std::cout << "\t\t\t remaining bits " << nAvailableBits <<  std::endl;
              if (nAvailableBits == 0) {
                  iLine++;
                  nAvailableBits = N_BITS_PER_WORD;
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
              bitsToRead = SS_CLUSTER_BITS - nAvailableBits;
              for (int i = 0; i < bitsToRead; i ++) { 
                nMask_newLine |= (1 << i);
              }
              // shift and mask
              uint16_t wordRight = (lines[iLine+1] >> (N_BITS_PER_WORD - bitsToRead)) & nMask_newLine; 
              // compose the full cluster word
              stripClustersWords[icluster] = ((wordLeft << bitsToRead) | wordRight);  
//               std::cout << "\t\t cluster " << icluster << "  on 2 lines \t clusterword " << std::bitset<SS_CLUSTER_BITS>(stripClustersWords[icluster]) <<  std::endl;
              
              // reset n available bits
              nAvailableBits = N_BITS_PER_WORD - bitsToRead ;
//               std::cout << "\t\t\t remaining bits " << nAvailableBits <<  std::endl;
              // advance by one line and re-init the number of complete clusters read from this line
              iLine++;
              nFullClusters = 0;
            }
          } // end loop storing cluster words

          // now create the Phase2TrackerCluster1D object 
          int count_clusters = 0;
          for (auto icluster : stripClustersWords){
            uint32_t chipID = (icluster >> 11) & CHIP_ID_MAX_VALUE;

            uint32_t sclusterAddress_toDelete = (icluster >> 3) & 0xFF; // only for debugging
            uint32_t sclusterAddress = (icluster >> 4) & SCLUSTER_ADDRESS_BITS_HEX; // why not uint16?
            bool isSeedSensor = (icluster >> 3) & IS_SEED_SENSOR_BITS;
            uint32_t width = icluster &  WIDTH_MAX_VALUE;
            // cluster width is truncated during packing (3 bits)
            // since width = 0 is unphysical, we can at least recover cluster with width == 8 
            // by assuming that clusters packed with width == 0 had in reality width = 8
            // this is a tmp fix, we should maybe think about how to properly do this.
            // also, for original widths > 8: again, due to truncation, they get an incorrect width of 
            // cluster.getWidth() & WIDTH_MAX_VALUE. should be probably fixed in the packer 
            // (e.g. if width > 8, pack with width = 0) 
            if (width == 0) width = 8;
            std::cout << "\t[unpacking] chipID : " <<  (chipID) << "\t " << std::bitset<3>(chipID) <<   std::endl;
            std::cout << "\t[unpacking] address : " << (sclusterAddress_toDelete) << "\t " << std::bitset<8>(sclusterAddress_toDelete) <<   std::endl;
            std::cout << "\t[unpacking] width : " << (width)   << "\t " << std::bitset<3>(width) <<   std::endl;
//             std::cout << "\t[unpacking] address minus seed : " << (sclusterAddress) << "\t " << std::bitset<7>(sclusterAddress) <<   std::endl;
            std::cout <<  std::endl;
            
            // now, rebasing to PR3
            unsigned int x = STRIPS_PER_CBC * chipID + sclusterAddress;
            unsigned int y = iChannel%2 == 0 ? 0 : 1; 
//             std::cout << "\t\t cluster#" << count_clusters << "\t x y width : " << x << " " << y << " " << width << "  is seed:" << isSeedSensor;

            Phase2TrackerCluster1D thisCluster = Phase2TrackerCluster1D(x, y, width);
//             std::cout << "\t\t check: width from Phase2TrackerCluster1D: " << thisCluster.size() << std::endl;
            if (isSeedSensor)
              thisChannel1DSeedClusters.push_back(thisCluster);
            else  
              thisChannel1DCorrClusters.push_back(thisCluster);
            count_clusters++;
            
          } // end loop on cluster words 
          
          if (count_clusters == 0 ) continue;
  
          // put the filler here, for the output collection
          // fill every 2 channels are read
          if (iChannel%2 != 1) 
            continue;
          
          // need the detid of this module
          // first I need to construct the DTCElinkId object ## dtc_id, gbtlink_id, elink_id
          // to get the gbt_id I should reverse what is done in the DTCUnit.convertToRawData function,
          // where clusters from channel X are split into 2*i and 2*i+1 based on being from CIC0 or CIC1
          unsigned int gbt_id = iSlink * MODULES_PER_SLINK + std::div(iChannel, 2).quot;
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
              // inner detid is defined as module detid + 1. First int in the pair from the map
//               std::cout << "\t\t -> detId:" <<  possibleDetIds->second << "  from map: " << stackMap_[possibleDetIds->second].first << std::endl;
              edmNew::DetSetVector<Phase2TrackerCluster1D>::FastFiller spcs(*outputClusterCollection, stackMap_[possibleDetIds->second].first);
              for (it = thisChannel1DSeedClusters.begin(); it != thisChannel1DSeedClusters.end(); it++) {
                spcs.push_back(*it);
              }
            }
            {
              // outer detid is defined as inner detid + 1 or module detid + 2. Second int in the pair from the map
//               std::cout << "\t\t -> detId:" <<  possibleDetIds->second << "  from map: " << stackMap_[possibleDetIds->second].second << std::endl;
              edmNew::DetSetVector<Phase2TrackerCluster1D>::FastFiller spcc(*outputClusterCollection, stackMap_[possibleDetIds->second].second);
              for (it = thisChannel1DCorrClusters.begin(); it != thisChannel1DCorrClusters.end(); it++) {
                spcc.push_back(*it);
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
    return channelIdx + N_BYTES_PER_WORD + iline * N_BYTES_PER_WORD; 
}

uint32_t RawToClusterProducer::readLine(const unsigned char* dataPtr, int lineIdx){
    uint32_t line = (static_cast<uint32_t>(dataPtr[lineIdx]) << 24) | 
                    (static_cast<uint32_t>(dataPtr[lineIdx + 1]) << 16) | 
                    (static_cast<uint32_t>(dataPtr[lineIdx + 2]) << 8) | 
                    (static_cast<uint32_t>(dataPtr[lineIdx + 3]));

    return line;                                
}                                


DEFINE_FWK_MODULE(RawToClusterProducer);