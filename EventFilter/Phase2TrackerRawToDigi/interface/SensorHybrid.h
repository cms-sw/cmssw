#ifndef SENSOR_HYBRID_H
#define SENSOR_HYBRID_H

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"

#include <bitset>

class SensorHybrid
{
    private:

        std::vector<Phase2TrackerCluster1D*> get_clusters_on_cic(edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator clusterIterator, const bool& cic_id, const TrackerGeometry& trackerGeometry, const int internal_id) 
        {
            using namespace Phase2TrackerSpecifications;
            using namespace Phase2DAQFormatSpecification;

            const GeomDetUnit* sensor_unit = trackerGeometry.idToDetUnit(clusterIterator->detId());
            unsigned int cic_boundary_in_z = CIC_Z_BOUNDARY_STRIPS;

            if (sensor_unit) 
            {
                TrackerGeometry::ModuleType moduleType = trackerGeometry.getDetectorType(clusterIterator->detId()); 
                switch (moduleType)
                {
                    case TrackerGeometry::ModuleType::Ph2PSS:
                        if (internal_id == 1)
                        {
                            sensor_type_1 = TrackerGeometry::ModuleType::Ph2PSS;
                        }
                        else if (internal_id == 2)
                        {
                            sensor_type_2 = TrackerGeometry::ModuleType::Ph2PSS;
                        }
                        (*this).number_of_strip_clusters_ = (*clusterIterator).size();
                        (*this).num_channels_per_chip = CHANNELS_PER_SSA;
                        break;

                    case TrackerGeometry::ModuleType::Ph2SS:
                        if (internal_id == 1)
                        {
                            sensor_type_1 = TrackerGeometry::ModuleType::Ph2SS;
                        }
                        else if (internal_id == 2)
                        {
                            sensor_type_2 = TrackerGeometry::ModuleType::Ph2SS;
                        }
                        (*this).number_of_strip_clusters_ = (*clusterIterator).size();
                        (*this).num_channels_per_chip = CHANNELS_PER_CBC;
                        break;

                    case TrackerGeometry::ModuleType::Ph2PSP:
                        if (internal_id == 1)
                        {
                            sensor_type_1 = TrackerGeometry::ModuleType::Ph2PSP;
                        }
                        else if (internal_id == 2)
                        {
                            sensor_type_2 = TrackerGeometry::ModuleType::Ph2PSP;
                        }
                        (*this).number_of_pixel_clusters_ = (*clusterIterator).size();
                        cic_boundary_in_z = CIC_Z_BOUNDARY_PIXEL; 
                        (*this).num_channels_per_chip = CHANNELS_PER_SSA;
                        break;

                    default:
                        throw cms::Exception("InvalidModuleType") 
                            << "Unexpected TrackerGeometry::ModuleType for detId: " 
                            << clusterIterator->detId() << ".";
                }
            }

            std::vector<Phase2TrackerCluster1D*> filteredClusters;

            if (clusterIterator != edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator{}) 
            {
                for (auto& cluster : *clusterIterator) 
                {
                    if ((cic_id == true && cluster.column() > cic_boundary_in_z))
                    {
                        filteredClusters.push_back(&cluster);
                    }
                    else if (cic_id == false && cluster.column() <= cic_boundary_in_z)
                    {
                        filteredClusters.push_back(&cluster);
                    }
                }
            }

            return filteredClusters;
        }

        void set_channel_cluster_payload(std::vector<Phase2DAQFormatSpecification::Word32Bits>& payload)
        {

            using namespace Phase2DAQFormatSpecification;
            using namespace Phase2TrackerSpecifications;

            uint32_t currentWord = 0;
            int bitsFilled = 0;

            // For PS, sensor_2 is always strip and sensor_1 is always pixel

            for (auto& cluster : sensor_2_clusters_) 
            {
                // cluster info 

                uint32_t chipID = std::div(cluster->firstStrip() * 2.0, num_channels_per_chip).quot & CHIP_ID_MAX_VALUE;       // 3 bits
                uint32_t sclusterAddress = std::div(cluster->firstStrip() * 2.0, num_channels_per_chip).rem & SCLUSTER_ADDRESS_MAX_VALUE;  // 8 bits
                uint32_t width = cluster->size() & WIDTH_MAX_VALUE;                       // 3 bits

                uint32_t clusterData = (chipID << (SS_CLUSTER_BITS - CHIP_ID_BITS)) | 
                                        (sclusterAddress << (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS)) | width;

                if (bitsFilled + SS_CLUSTER_BITS <= NUMBER_OF_BITS_PER_WORD) 
                {
                    currentWord |= clusterData << (NUMBER_OF_BITS_PER_WORD - bitsFilled - SS_CLUSTER_BITS);
                    bitsFilled += SS_CLUSTER_BITS;
                } else 
                {
                    int bitsLeft = NUMBER_OF_BITS_PER_WORD - bitsFilled;
                    currentWord |= clusterData >> (SS_CLUSTER_BITS - bitsLeft);
                    payload.push_back(currentWord);

                    currentWord = clusterData << (NUMBER_OF_BITS_PER_WORD - (SS_CLUSTER_BITS - bitsLeft));
                    bitsFilled = SS_CLUSTER_BITS - bitsLeft;
                }
            }

            // std::cout << "Starting pixel clusters" << std::endl;
            
            for (auto& cluster : sensor_1_clusters_) 
            {
                // cluster info 

                uint32_t chipID = std::div(cluster->firstStrip() * 2.0, num_channels_per_chip).quot & CHIP_ID_MAX_VALUE;       // 3 bits
                uint32_t sclusterAddress = std::div(cluster->firstStrip() * 2.0, num_channels_per_chip).rem & SCLUSTER_ADDRESS_MAX_VALUE;  // 8 bits
                uint32_t width = cluster->size() & WIDTH_MAX_VALUE;                       // 3 bits

                uint32_t clusterData = (chipID << (SS_CLUSTER_BITS - CHIP_ID_BITS)) | 
                                        (sclusterAddress << (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS)) | width;

                if (bitsFilled + SS_CLUSTER_BITS <= NUMBER_OF_BITS_PER_WORD) 
                {
                    currentWord |= clusterData << (NUMBER_OF_BITS_PER_WORD - bitsFilled - SS_CLUSTER_BITS);
                    bitsFilled += SS_CLUSTER_BITS;
                } else 
                {
                    int bitsLeft = NUMBER_OF_BITS_PER_WORD - bitsFilled;
                    currentWord |= clusterData >> (SS_CLUSTER_BITS - bitsLeft);
                    payload.push_back(currentWord);

                    currentWord = clusterData << (NUMBER_OF_BITS_PER_WORD - (SS_CLUSTER_BITS - bitsLeft));
                    bitsFilled = SS_CLUSTER_BITS - bitsLeft;
                }
            }

            if (bitsFilled > 0) 
            {
                payload.push_back(currentWord);
            }
        }

        bool cic_id_;

        std::vector<Phase2TrackerCluster1D*> sensor_1_clusters_; // always pixel in the case of Phase2PS
        TrackerGeometry::ModuleType sensor_type_1;
        std::vector<Phase2TrackerCluster1D*> sensor_2_clusters_; // will be strip in the case of Phase2PS
        TrackerGeometry::ModuleType sensor_type_2;
        int num_channels_per_chip;

        unsigned int offset_index_;
        unsigned int number_of_pixel_clusters_ = 0;
        unsigned int number_of_strip_clusters_ = 0;
        unsigned int eventId_ = 0;

    public:

        SensorHybrid(edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator sensor_1, 
                    edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator sensor_2, const bool cic_id, const TrackerGeometry& trackerGeometry, const unsigned int eventId) : cic_id_(cic_id), eventId_(eventId)
        {
            sensor_1_clusters_ = get_clusters_on_cic(sensor_1, cic_id, trackerGeometry, 1);
            sensor_2_clusters_ = get_clusters_on_cic(sensor_2, cic_id, trackerGeometry, 2);
        }

        unsigned int    get_number_of_pixel_clusters() const { return number_of_pixel_clusters_; }
        unsigned int    get_number_of_strip_clusters() const { return number_of_strip_clusters_; }
        unsigned int    get_payload_size() const 
        { 
            using namespace Phase2DAQFormatSpecification;
            return (number_of_strip_clusters_ * SS_CLUSTER_BITS + number_of_pixel_clusters_ * PX_CLUSTER_BITS) / NUMBER_OF_BITS_PER_WORD;
        }
        unsigned int    get_offset_within_payload() const;
        const bool      get_cic_id() const { return cic_id_; }
        unsigned int    get_module_slink_id() const;
        unsigned int    get_module_slink_mod_id() const;
        std::vector<Phase2TrackerCluster1D*> get_sensor_1_clusters() const { return sensor_1_clusters_; }
        std::vector<Phase2TrackerCluster1D*> get_sensor_2_clusters() const { return sensor_2_clusters_; }
        TrackerGeometry::ModuleType get_sensor_type_1() const { return sensor_type_1; }
        TrackerGeometry::ModuleType get_sensor_type_2() const { return sensor_type_2; }

        void set_payload(std::vector<Phase2DAQFormatSpecification::Word32Bits>& payload)
        {
            using namespace Phase2DAQFormatSpecification;

            // Extracting values
            uint32_t eventID = eventId_ & L1ID_MAX_VALUE;  // 9 bits for eventId_
            uint32_t channelErrors = 0;  // 9 bits for errors, set to 0
            uint32_t num_strip_clusters = number_of_strip_clusters_ & ((1 << N_STRIP_CLUSTER_BITS) - 1); // 7 bits for strip clusters
            uint32_t num_pixel_clusters = number_of_pixel_clusters_ & ((1 << N_PIXEL_CLUSTER_BITS) - 1); // 7 bits for pixel clusters

            // Combine fields into the 32-bit header
            uint32_t header_ = (eventID << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS)) |
                            (channelErrors << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS)) |
                            (num_strip_clusters << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - N_STRIP_CLUSTER_BITS)) |
                            (num_pixel_clusters);

            // Convert to Word32Bits and add to payload
            Word32Bits header(header_);
            payload.push_back(header);
            set_channel_cluster_payload(payload);

            for (auto& i : payload)
            {
                std::cout << i.to_string() << std::endl;
            }

        }

};

#endif