#ifndef EventFilter_Phase2TrackerRawToDigi_ChannelsMask_H
#define EventFilter_Phase2TrackerRawToDigi_ChannelsMask_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerSpecifications.h"

#include <cstdint>
#include <array>
#include <cstdio>

/**
 * @brief: ChannelsMask Object, decodes the mask present in a single FED.
 */

class ChannelsMask {
public:

    ChannelsMask() : words_{{0, 0}}, globalMask_(0) {}
    
    explicit ChannelsMask(const std::array<uint32_t, 2>& words) 
        : words_(words), globalMask_(computeGlobalMask()) {}
    
    ChannelsMask(uint32_t w0, uint32_t w1) 
        : words_{{w0, w1}}, globalMask_(computeGlobalMask()) {}

    /**
     * @brief Print the channels mask in hexadecimal format
     */
    void print() const {
        printf("0x%08X %08X\n", 
                (unsigned int)words_[0], 
                (unsigned int)words_[1]);
    }

    /**
     * @brief Get Channel Mask #0 (bits 17-0 of word0)
     * @return 18-bit mask for channel 0
     */
    uint32_t getChannelMask0() const {
        return words_[0] & 0x3FFFF;  // 18 bits (bits 17-0)
    }

    /**
     * @brief Get Channel Mask #1 (bits 17-0 of word1)
     * @return 18-bit mask for channel 1
     */
    uint32_t getChannelMask1() const {
        return words_[1] & 0x3FFFF;  // 18 bits (bits 17-0)
    }

    /**
     * @brief Get the global 36-bit mask combining both channels
     * @return 36-bit mask where bits 35-18 = mask1, bits 17-0 = mask0
     */
    uint64_t getGlobalMask() const {
        return globalMask_;
    }

    /**
     * @brief Check if a specific channel is masked in the global mask register
     * @param channel Channel number (0-35)
     * @return true if channel is masked, false otherwise
     */
    bool isChannelMasked(int channel) const {
        if (channel < 0 || channel > Phase2TrackerSpecifications::CICs_PER_SLINK - 1) {
            throw cms::Exception("InvalidChannel") << "Channel number must be between 0 and " << Phase2TrackerSpecifications::CICs_PER_SLINK - 1 << ".";
        }
        return (globalMask_ >> channel) & 0x1ULL;
    }

    /**
     * @brief Check if a specific channel is enabled in the global mask register
     * @param channel Channel number (0-35)
     * @return true if channel is enabled, false otherwise
     */
    bool isChannelEnabled(int channel) const {
        if (channel < 0 || channel > Phase2TrackerSpecifications::CICs_PER_SLINK - 1) {
            throw cms::Exception("InvalidChannel") << "Channel number must be between 0 and " << Phase2TrackerSpecifications::CICs_PER_SLINK - 1 << ".";
        }
        return !isChannelMasked(channel);
    }

    /**
     * @brief Print a summary of which channels are enabled
     */
    void printSummary() const {
        printf("Global Mask (36 bits): 0x%09llX (", (unsigned long long)globalMask_);
        for (int i = 35; i >= 0; --i) {
            printf("%d", (int)((globalMask_ >> i) & 1ULL));  // Cast to int
            if (i % 4 == 0 && i != 0) printf(" ");
        }
        printf(")\n");
    }

private:
    std::array<uint32_t, 2> words_;  // 2 x 32-bit = 64-bit channels mask
    uint64_t globalMask_;            // 36-bit combined mask (computed at construction)
    
    /**
     * @brief Compute the global 36-bit mask from the two words
     * @return 36-bit mask combining both channel masks
     */
    uint64_t computeGlobalMask() const {
        uint32_t mask0 = words_[0] & 0x3FFFF;
        uint32_t mask1 = words_[1] & 0x3FFFF;
        return (static_cast<uint64_t>(mask1) << 18) | mask0;
    }
};

#endif