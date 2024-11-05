#ifndef DTCUNIT_H
#define DTCUNIT_H

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/Phase2TrackerRawToDigi/interface/Cluster.h"
#include <fstream>

class DTCUnit
{
    public:

        enum DTCType { PS, S2, Unknown };

        // initialize SLinkCollection with 4 FEDRawData objects
        DTCUnit(unsigned int& event) : ClusterCollection(4, std::vector<std::vector<Cluster>>(18)), SLinkCollection(4), Type(DTCType::Unknown), EventID(event) 
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
        FEDRawData& GetSLink(const unsigned int& SLinkID) 
        {
            // if SLinkID is out of range [0, 4], throw cms exception
            if (SLinkID > 3)
            {
                throw cms::Exception("DTCUnit") << "SLinkID " << SLinkID << " is out of range [0, 4)";
            }
            else
            {
                return SLinkCollection[SLinkID];
            }
        }

        ~DTCUnit() {}

        // Method to determine DTCType based on ClusterCollection
        DTCType getDTCType() 
        {
            // Iterate over the ClusterCollection to find the first matching cluster type
            for (auto& clusterVector : ClusterCollection)
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
            return ClusterCollection[index];
        }

        std::vector<FEDRawData>& getSLinks()
        {
            return SLinkCollection;
        }

        void convertToRawData(const std::size_t index)
        {
            std::vector<std::vector<Cluster>>& SLinks_0 = ClusterCollection[index];
            std::vector<std::vector<Cluster>> newClusterCollection(36);

            // Organize clusters into CIC-0 and CIC-1 clusters
            for (size_t i = 0; i < SLinks_0.size(); ++i) {
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

                newClusterCollection[2 * i] = cicId_0_clusters;
                newClusterCollection[2 * i + 1] = cicId_1_clusters;
            }

            // We will store the final 32-bit words (both offset map and payload)
            std::vector<uint32_t> words;

            // fill the first 4 words with the header with all 1s
            words.push_back(0xFFFFFFFF);
            words.push_back(0xFFFFFFFF);
            words.push_back(0xFFFFFFFF);
            words.push_back(0xFFFFFFFF);

            // --- Step 1: Create the offset map ---
            std::vector<uint16_t> offsetMap(36, 0);  // 36 channels, initially 0
            uint32_t currentOffset = 0;              // Start after the 4 words of the header

            // --- Step 2: Build the payload ---
            std::vector<uint32_t> payload;

            // Iterate through each channel and prepare the payload
            for (size_t channel_index = 0; channel_index < newClusterCollection.size(); ++channel_index) 
            {
                std::vector<Cluster>& clusters = newClusterCollection[channel_index];
                // if (clusters.empty()) continue;

                // Create the 32-bit header word for the channel
                uint32_t eventID = EventID & 0x1FF;  // EventID (9 bits)
                uint32_t channelErrors = 0;          // 9 bits for errors, set to 0
                uint32_t numClusters = clusters.size();

                // std::cout << index << ", " << numClusters << std::endl;

                // Build the channel header
                uint32_t header = (eventID << 23) | (channelErrors << 14) | (numClusters << 7);

                // Push the header into the payload
                payload.push_back(header);

                // Update the offset map
                offsetMap[channel_index] = currentOffset;
                currentOffset++;

                // Now pack the clusters into 32-bit words
                uint32_t currentWord = 0;
                int bitsFilled = 0;
                if (clusters.size() > 0)
                  std::cout << "[packing] n clusters for channel " << channel_index   << ": " << clusters.size() << std::endl;

                for (auto& cluster : clusters) 
                {
                    // cluster info 

                    uint32_t chipID = cluster.getChipId() & 0x7;              // 3 bits
                    uint32_t sclusterAddress = cluster.getSclusterAddress() & 0xFF;  // 8 bits
                    uint32_t width = cluster.getWidth() & 0x7;                // 3 bits

                    std::cout << "[packing] chipID : " <<  cluster.getChipId() <<  std::endl;
                    std::cout << "[packing] address : " << cluster.getSclusterAddress() <<  std::endl;
                    std::cout << "[packing] width : " << cluster.getWidth()  <<  std::endl;
                    std::cout <<  std::endl;

                    uint32_t clusterData = (chipID << 11) | (sclusterAddress << 3) | width;

                    if (bitsFilled + 14 <= 32) {
                        currentWord |= clusterData << (32 - bitsFilled - 14);
                        bitsFilled += 14;
                    } else {
                        int bitsLeft = 32 - bitsFilled;
                        currentWord |= clusterData >> (14 - bitsLeft);
                        payload.push_back(currentWord);  // Push full word
                        currentOffset++;

                        currentWord = clusterData << (32 - (14 - bitsLeft));
                        bitsFilled = 14 - bitsLeft;
                    }
                }

                if (bitsFilled > 0) {
                    payload.push_back(currentWord);  // Push any remaining bits
                    currentOffset++;
                }
            }


            // --- Step 3: Combine the offset map into 32-bit words ---
            for (size_t idx = 0; idx < offsetMap.size(); idx += 2) {
//                 std::cout << "[packing] offsetMap[" << idx   << "]: " << offsetMap[idx] << std::endl;
//                 std::cout << "          offsetMap[" << idx+1 << "]: " << offsetMap[idx+1] << std::endl;
                uint32_t word = (offsetMap[idx + 1] << 16) | offsetMap[idx];
                words.push_back(word);  // Push offset map word
            }

            // --- Step 4: Add payload words ---
            for (auto& word : payload) {
                words.push_back(word);
            }

            // --- Step 5: Fill `data_ptr` with the computed 32-bit words ---
            FEDRawData& RawDataOnSLink = SLinkCollection[index];
            RawDataOnSLink.resize(words.size() * 4, 4);  // Resize the buffer to fit all 32-bit words
            unsigned char *data_ptr = RawDataOnSLink.data();

            for (size_t word_index = 0; word_index < words.size(); ++word_index) {
                insertHexWordAt(data_ptr, word_index, words[word_index]);
            }

            size_t actual_used_bytes = words.size() * 4;  // Total size used
            RawDataOnSLink.resize(actual_used_bytes, 4);  
        }


    private:
        // Each SLink handles 18 Modules (36 CICs) and each DTC has 4 of those
        // So, each DTC handles 72 Modules (144 CICs)

        std::vector<std::vector<std::vector<Cluster>>> ClusterCollection;
        std::vector<FEDRawData> SLinkCollection;

        DTCType Type;
        unsigned int EventID;
};

#endif
