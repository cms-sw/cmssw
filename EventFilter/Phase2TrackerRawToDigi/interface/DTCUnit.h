#ifndef DTCUNIT_H
#define DTCUNIT_H

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/Phase2TrackerRawToDigi/interface/Cluster.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include <fstream>

/**
 * @class DTCUnit
 * @brief Class to represent a single DTC unit for the phase 2 tracker
 */

class DTCUnit
{
    public:

        enum DTCType { PS, S2, Unknown };

        // initialize sLinkCollection_ with 4 FEDRawData objects
        DTCUnit(unsigned int& event) : clusterCollection_(4, std::vector<std::vector<Cluster>>(18)), sLinkCollection_(4), Type(DTCType::Unknown), eventId_(event) 
        {
            
        }

        void insertHexWordAt(unsigned char *data_ptr, size_t word_index, uint32_t hex_word) 
        {
            data_ptr[word_index * 4 + 0] = (hex_word >> 24) & 0xFF;  // Most significant byte (bits 31-24)
            data_ptr[word_index * 4 + 1] = (hex_word >> 16) & 0xFF;  // Next byte (bits 23-16)
            data_ptr[word_index * 4 + 2] = (hex_word >> 8) & 0xFF;   // Next byte (bits 15-8)
            data_ptr[word_index * 4 + 3] = (hex_word >> 0) & 0xFF;   // Least significant byte (bits 7-0)
        }

        // GetSLink Method
        FEDRawData& getSLink(const unsigned int& SLinkID) 
        {
            // if SLinkID is out of range [0, 4], throw cms exception
            if (SLinkID > Phase2TrackerSpecifications::SLINKS_PER_DTC - 1)
            {
                throw cms::Exception("DTCUnit") << "SLinkID " << SLinkID << " is out of range [0, 4)";
            }
            else
            {
                return sLinkCollection_[SLinkID];
            }
        }

        ~DTCUnit() {}

        // Method to determine DTCType based on clusterCollection_
        DTCType getDTCType() 
        {
            // Iterate over the clusterCollection_ to find the first matching cluster type
            for (auto& clusterVector : clusterCollection_)
            {
                for (auto& cluster_ : clusterVector)
                {
                    for (auto& cluster : cluster_)
                    {
                        // Check if the cluster type is Ph2PSP or Ph2PSS
                        if (cluster.getClusterType() == TrackerGeometry::ModuleType::Ph2PSP || 
                            cluster.getClusterType() == TrackerGeometry::ModuleType::Ph2PSS)
                        {
                            return DTCType::PS;
                        }
                        // Check if the cluster type is Ph2SS
                        else if (cluster.getClusterType() == TrackerGeometry::ModuleType::Ph2SS)
                        {
                            return DTCType::S2;
                        }
                    }
                }
            }

                        // If no matching cluster type is found, return Unknown
            return DTCType::Unknown;
        }

        std::vector<std::vector<Cluster>>& getClustersOnSLink(const int& index)
        {
            return clusterCollection_[index];
        }

        std::vector<FEDRawData>& getSLinks()
        {
            return sLinkCollection_;
        }

        void convertToRawData(const std::size_t index)
        {

            using namespace Phase2TrackerSpecifications;
            using namespace Phase2DAQFormatSpecification;

            std::vector<std::vector<Cluster>>& SLinks_0 = clusterCollection_[index];
            std::vector<std::vector<Cluster>> newclusterCollection_(36);

            // Organize clusters into CIC-0 and CIC-1 clusters
            for (size_t i = 0; i < SLinks_0.size(); ++i) 
            {
                std::vector<Cluster>& clusters = SLinks_0[i];
                std::vector<Cluster> cicId_0_clusters;
                std::vector<Cluster> cicId_1_clusters;

                for (auto& cluster : clusters) {
                    if (cluster.getCicId() == 0) {
                        cicId_0_clusters.push_back(cluster);
                    } else if (cluster.getCicId() == 1) {
                        cicId_1_clusters.push_back(cluster);
                    }
                }

                newclusterCollection_[2 * i] = cicId_0_clusters;
                newclusterCollection_[2 * i + 1] = cicId_1_clusters;

            }

            // We will store the final 32-bit words (both offset map and payload)
            std::vector<uint32_t> words;

            // fill the first 4 words with the header with all 1s
            words.push_back(DTC_DAQ_HEADER);
            words.push_back(DTC_DAQ_HEADER);
            words.push_back(DTC_DAQ_HEADER);
            words.push_back(DTC_DAQ_HEADER);

            // --- Step 1: Create the offset map ---
            std::vector<uint16_t> offsetMap(CICs_PER_SLINK, 0);  // 36 channels, initially 0
            uint32_t currentOffset = 0;              // Start after the 4 words of the header

            // --- Step 2: Build the payload ---
            std::vector<uint32_t> payload;

            // Iterate through each channel and prepare the payload
            for (size_t channel_index = 0; channel_index < newclusterCollection_.size(); ++channel_index) 
            {
                std::vector<Cluster>& clusters = newclusterCollection_[channel_index];

                // Create the 32-bit header word for the channel
                uint32_t eventID = eventId_ & L1ID_MAX_VALUE;  // eventId_ (9 bits)
                uint32_t channelErrors = 0;          // 9 bits for errors, set to 0
                uint32_t numClusters = clusters.size();

                // Build the channel header
                uint32_t header = (eventID << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS)) | 
                                  (channelErrors << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS)) | 
                                  (numClusters << (NUMBER_OF_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - NCLUSTERS_BITS));

                // Push the header into the payload
                payload.push_back(header);

                // Update the offset map
                offsetMap[channel_index] = currentOffset;
                currentOffset++;

                // Now pack the clusters into 32-bit words
                uint32_t currentWord = 0;
                int bitsFilled = 0;

                for (auto& cluster : clusters) 
                {
                    // cluster info 

                    uint32_t chipID = cluster.getChipId() & CHIP_ID_MAX_VALUE;       // 3 bits
                    uint32_t sclusterAddress = cluster.getSclusterAddress() & SCLUSTER_ADDRESS_MAX_VALUE;  // 8 bits
                    uint32_t width = cluster.getWidth() & WIDTH_MAX_VALUE;                       // 3 bits

                    uint32_t clusterData = (chipID << (SS_CLUSTER_BITS - CHIP_ID_BITS)) | 
                                            (sclusterAddress << (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS)) | width;

                    if (bitsFilled + SS_CLUSTER_BITS <= NUMBER_OF_BITS_PER_WORD) {
                        currentWord |= clusterData << (NUMBER_OF_BITS_PER_WORD - bitsFilled - SS_CLUSTER_BITS);
                        bitsFilled += SS_CLUSTER_BITS;
                    } else {
                        int bitsLeft = NUMBER_OF_BITS_PER_WORD - bitsFilled;
                        currentWord |= clusterData >> (SS_CLUSTER_BITS - bitsLeft);
                        payload.push_back(currentWord);  // Push full word
                        currentOffset++;

                        currentWord = clusterData << (NUMBER_OF_BITS_PER_WORD - (SS_CLUSTER_BITS - bitsLeft));
                        bitsFilled = SS_CLUSTER_BITS - bitsLeft;
                    }
                }

                if (bitsFilled > 0) 
                {
                    payload.push_back(currentWord);  // Push any remaining bits
                    currentOffset++;
                }
            }

            // --- Step 3: Combine the offset map into 32-bit words ---
            for (size_t idx = 0; idx < offsetMap.size(); idx += 2) 
            {
                uint32_t word = (offsetMap[idx + 1] << NUMBER_OF_BITS_PER_WORD / 2) | offsetMap[idx];
                words.push_back(word);  // Push offset map word
            }

            // --- Step 4: Add payload words ---
            for (auto& word : payload) { words.push_back(word); }

            // --- Step 5: Fill `data_ptr` with the computed 32-bit words ---
            FEDRawData& RawDataOnSLink = sLinkCollection_[index];
            RawDataOnSLink.resize(words.size() * NUMBER_OF_BYTES_PER_WORD, NUMBER_OF_BYTES_PER_WORD);  // Resize the buffer to fit all 32-bit words
            unsigned char *data_ptr = RawDataOnSLink.data();

            for (size_t word_index = 0; word_index < words.size(); ++word_index) 
            {
                insertHexWordAt(data_ptr, word_index, words[word_index]);
            }

            size_t actual_used_bytes = words.size() * NUMBER_OF_BYTES_PER_WORD;  // Total size used
            RawDataOnSLink.resize(actual_used_bytes, NUMBER_OF_BYTES_PER_WORD);  
        }

        void convertToRawData()
        {
            for (size_t i = 0; i < sLinkCollection_.size(); ++i)
            {
                convertToRawData(i);
            }
        }


    private:
        // Each SLink handles 18 Modules (36 CICs) and each DTC has 4 of those
        // So, each DTC handles 72 Modules (144 CICs)

        std::vector<std::vector<std::vector<Cluster>>> clusterCollection_;
        std::vector<FEDRawData> sLinkCollection_;

        DTCType Type;
        unsigned int eventId_;
};

#endif
