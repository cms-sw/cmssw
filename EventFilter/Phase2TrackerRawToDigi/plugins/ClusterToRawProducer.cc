#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

// #include "EventFilter/Phase2TrackerRawToDigi/interface/DTCAssembly.h"
// #include "EventFilter/Phase2TrackerRawToDigi/interface/Cluster.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/SensorHybrid.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include <fstream>

class ClusterToRawProducer : public edm::one::EDProducer<> {
public:
    explicit ClusterToRawProducer(const edm::ParameterSet&);
    ~ClusterToRawProducer() override;

private:
    void produce(edm::Event&, const edm::EventSetup&) override;
    
    edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> ClusterCollectionToken_;
    const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryToken_;
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyToken_;

    void insertHexWordAt(unsigned char *data_ptr, size_t word_index, uint32_t hex_word) 
    {
        data_ptr[word_index * 4 + 0] = (hex_word >> 24) & 0xFF;  // Most significant byte (bits 31-24)
        data_ptr[word_index * 4 + 1] = (hex_word >> 16) & 0xFF;  // Next byte (bits 23-16)
        data_ptr[word_index * 4 + 2] = (hex_word >> 8) & 0xFF;   // Next byte (bits 15-8)
        data_ptr[word_index * 4 + 3] = (hex_word >> 0) & 0xFF;   // Least significant byte (bits 7-0)
    }

};

ClusterToRawProducer::ClusterToRawProducer(const edm::ParameterSet& iConfig)
    : ClusterCollectionToken_(consumes<Phase2TrackerCluster1DCollectionNew>(iConfig.getParameter<edm::InputTag>("Phase2Clusters"))),
      cablingMapToken_(esConsumes()),
      trackerGeometryToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>())
{
    produces<FEDRawDataCollection>();
}

ClusterToRawProducer::~ClusterToRawProducer() { }

void ClusterToRawProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
    // Retrieve TrackerGeometry and TrackerTopology from EventSetup
    const TrackerGeometry& trackerGeometry = iSetup.getData(trackerGeometryToken_);
    const TrackerTopology& trackerTopology = iSetup.getData(trackerTopologyToken_);
    
    // Retrieve the CablingMap
    const auto& cablingMap = iSetup.getData(cablingMapToken_);

    // get EventID and RunID
    unsigned int eventId_ = iEvent.id().event();

    // Create FEDRawDataCollection to store the output
    auto fedRawDataCollection = std::make_unique<FEDRawDataCollection>();

    using namespace Phase2TrackerSpecifications;
    using namespace Phase2DAQFormatSpecification;

    for (int dtc_id = MIN_DTC_ID; dtc_id < MAX_DTC_ID; dtc_id++)
    {
        for (int slink_id = MIN_SLINK_ID; slink_id < MAX_SLINK_ID + 1; slink_id++)
        {
            int index_first = slink_id * MODULES_PER_SLINK;
            int index_last = (slink_id + 1) * MODULES_PER_SLINK;

            FEDRawData slink_daq_stream;

            std::vector<Word32Bits> daq_packet;
            std::vector<Word32Bits> offset_map(CICs_PER_SLINK / 2, Word32Bits(0));

            for (int i = 0; i < 4; ++i) { daq_packet.push_back(Word32Bits(DTC_DAQ_HEADER)); }
            std::vector<Word32Bits> payload;

            unsigned int offset_in_32b_words = 0;

            for (int module_id = index_first; module_id < index_last; module_id++) 
            {
                const unsigned int module_id_within_slink = module_id - index_first;
                DTCELinkId cms_link_id = DTCELinkId(dtc_id, module_id, 0);
                try 
                {
                    auto link_to_det_association = cablingMap.dtcELinkIdToDetId(cms_link_id);
                    const DTCELinkId& link_id = link_to_det_association->first;
                    const DetId& det_id = link_to_det_association->second;

                    edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator sensor_1_cluster_collection = iEvent.get(ClusterCollectionToken_).find(det_id + 1);
                    edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator sensor_2_cluster_collection = iEvent.get(ClusterCollectionToken_).find(det_id + 2);

                    // sensor_1_cic_0 and sensor_2_cic_0 form a single output daq channel.
                    SensorHybrid Hybrid_1 (sensor_1_cluster_collection, sensor_2_cluster_collection, 0, trackerGeometry, eventId_);

                    // // sensor_1_cic_1 and sensor_2_cic_1 form a single output daq channel.
                    SensorHybrid Hybrid_2 (sensor_1_cluster_collection, sensor_2_cluster_collection, 1, trackerGeometry, eventId_);

                    // Figure Out Offsets
                    uint16_t hybrid_1_offset = offset_in_32b_words;
                    offset_in_32b_words += Hybrid_1.get_payload_size();

                    uint16_t hybrid_2_offset = offset_in_32b_words;
                    offset_in_32b_words += Hybrid_2.get_payload_size();

                    std::cout << hybrid_1_offset << ", " << hybrid_2_offset << " | " << slink_id << " | " << module_id << std::endl;
                    std::cout << Hybrid_1.get_number_of_strip_clusters() << ", " << Hybrid_1.get_number_of_pixel_clusters() << std::endl;
                    std::cout << Hybrid_2.get_number_of_strip_clusters() << ", " << Hybrid_2.get_number_of_pixel_clusters() << std::endl;

                    // 24 is PSS, 23 is PSP, 26 is SS-SS
                    uint32_t combined_offsets = (static_cast<uint32_t>(hybrid_2_offset) << 16) | hybrid_1_offset;
                    offset_map[module_id_within_slink] = Word32Bits(combined_offsets);

                    // Figure out Payload
                    Hybrid_1.set_payload(payload);
                    Hybrid_2.set_payload(payload);
                } 
                catch (const cms::Exception& e) 
                {
                    // exception here means that the link is not connected to a detector
                    uint32_t eventID = eventId_ & L1ID_MAX_VALUE;  // eventId_ (9 bits)
                    uint32_t channelErrors = 0;  // 9 bits for errors, all set to 0
                    uint32_t numClusters = 0;    // no clusters here.

                    // Build the channel header
                    uint32_t header_ = (eventID << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS)) |
                            (channelErrors << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS)) |
                            (numClusters << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - N_STRIP_CLUSTER_BITS)) |
                            (numClusters);

                    uint16_t hybrid_1_offset = offset_in_32b_words;
                    offset_in_32b_words += 1;
                    
                    uint16_t hybrid_2_offset = offset_in_32b_words;
                    offset_in_32b_words += 1;

                    uint32_t combined_offsets = (static_cast<uint32_t>(hybrid_2_offset) << 16) | hybrid_1_offset;
                    offset_map[module_id_within_slink] = Word32Bits(combined_offsets);

                    // Push the header into the payload
                    payload.push_back(Word32Bits(header_));
                    payload.push_back(Word32Bits(header_));

                    // continue;
                }
            }

            // Add the offset map to the slink_daq_stream
            for (std::size_t i = 0; i < offset_map.size(); i++)
            { daq_packet.push_back(offset_map[i]); }

            // Add the payload to the slink_daq_stream
            for (std::size_t i = 0; i < payload.size(); i++)
            { daq_packet.push_back(payload[i]); }

            slink_daq_stream.resize(daq_packet.size() * NUMBER_OF_BYTES_PER_WORD, NUMBER_OF_BYTES_PER_WORD);  // Resize the buffer to fit all 32-bit words
            unsigned char *data_ptr = slink_daq_stream.data();

            for (size_t word_index = 0; word_index < daq_packet.size(); ++word_index)
            {
                insertHexWordAt(data_ptr, word_index, (daq_packet[word_index].to_ulong()));
            }

            size_t actual_used_bytes = daq_packet.size() * NUMBER_OF_BYTES_PER_WORD;  // Total size used
            slink_daq_stream.resize(actual_used_bytes, NUMBER_OF_BYTES_PER_WORD);  

            fedRawDataCollection.get()->FEDData( slink_id + SLINKS_PER_DTC * (dtc_id - 1) + TRACKER_HEADER ) = slink_daq_stream;

        }
    }

    iEvent.put(std::move(fedRawDataCollection));

}

DEFINE_FWK_MODULE(ClusterToRawProducer);