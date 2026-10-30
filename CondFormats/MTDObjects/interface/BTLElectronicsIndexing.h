#ifndef CondFormats_MTDObjects_BTLElectronicsIndexing_h
#define CondFormats_MTDObjects_BTLElectronicsIndexing_h

#include <cstdint>

#include "CondFormats/MTDObjects/interface/BTLElectronicsSpecs.h"

/**
 *  Dense indexing of the BTL electronics channels.
 *
 *  Plain constexpr functions - usable on host and in alpaka kernels.
 */

namespace BTLElectronicsIndexing {

  static constexpr int32_t nPairsPerChip =
      BTLElectronicsSpecs::nChannelsPerELink / 2;  // plus/minus channel pairs (crystals) per chip

  // total number of electronics channels and of chips (e-links) in BTL
  static constexpr int32_t kNumberOfChannels =
      (BTLElectronicsSpecs::kNumberOfFEDs / BTLElectronicsSpecs::nFEDsPerSerenity) *
      BTLElectronicsSpecs::nHSLinksPerSerenity * BTLElectronicsSpecs::nELinksPerHSLink *
      BTLElectronicsSpecs::nChannelsPerELink;                                                            // 331776
  static constexpr int32_t kNumberOfChips = kNumberOfChannels / BTLElectronicsSpecs::nChannelsPerELink;  // 10368

  // Dense index of the electronics channel (fedId, hs, el, channel), in [0, kNumberOfChannels).
  // The chip (e-link) index is denseChannelIndex / BTLElectronicsSpecs::nChannelsPerELink.
  //   fedId   FED id, assumed in [kFirstFEDId, kFirstFEDId + kNumberOfFEDs) (see BTLElectronicsSpecs)
  //   hs      HS-link id within the Serenity (0-35)
  //   el      e-link id within the HS-link (0-23)
  //   channel TOFHIR channel id within the e-link (0-31)
  constexpr int32_t denseChannelIndex(int32_t fedId, int32_t hs, int32_t el, int32_t channel) {
    const int32_t serenityIdx =
        (fedId - BTLElectronicsSpecs::kFirstFEDId) / BTLElectronicsSpecs::nFEDsPerSerenity;  // Serenity index [0-11]
    const int32_t hsIdx = serenityIdx * BTLElectronicsSpecs::nHSLinksPerSerenity + hs;       // HS-link index 
    const int32_t chipIdx = hsIdx * BTLElectronicsSpecs::nELinksPerHSLink + el;  // chip (e-link) index 
    return chipIdx * BTLElectronicsSpecs::nChannelsPerELink + channel;
  }

  // Checks that (hs, el) are within range and that the HS-link belongs to this FED:
  // FED k of a Serenity carries the HS-links [k * nHSLinksPerFED, (k + 1) * nHSLinksPerFED).
  // fedId is assumed in [kFirstFEDId, kFirstFEDId + kNumberOfFEDs) (see BTLElectronicsSpecs).
  constexpr bool isValidLink(int32_t fedId, int32_t hs, int32_t el) {
    const int32_t fedInSerenity = (fedId - BTLElectronicsSpecs::kFirstFEDId) % BTLElectronicsSpecs::nFEDsPerSerenity;
    return hs >= 0 && hs < BTLElectronicsSpecs::nHSLinksPerSerenity && el >= 0 &&
           el < BTLElectronicsSpecs::nELinksPerHSLink && hs / BTLElectronicsSpecs::nHSLinksPerFED == fedInSerenity;
  }

}  // namespace BTLElectronicsIndexing

#endif
