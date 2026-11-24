#ifndef CondFormats_MTDObjects_BTLElectronicsToDetIdMappingSoA_h
#define CondFormats_MTDObjects_BTLElectronicsToDetIdMappingSoA_h

#include <alpaka/alpaka.hpp>
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// -----------------------------------------------------------------------
// Dense electronics -> crystal rawId lookup
// Flat array used by the alpaka BTL unpacker on any backend
// The table is indexed by the dense channel index
// BTLElectronicsIndexing::denseChannelIndex(fedId, hsLinkId, eLinkId, channelId).
// -----------------------------------------------------------------------

GENERATE_SOA_LAYOUT(BTLElectronicsToDetIdMappingSoALayout,
                    SOA_COLUMN(uint32_t, rawId),
                    SOA_COLUMN(bool, valid),
                    SOA_COLUMN(uint8_t, partnerChannelId),  // TOFHIR channel of the other side of the same crystal
                    SOA_COLUMN(int8_t, side))               // side: 1 = plus, 0 = minus

using BTLElectronicsToDetIdMappingSoA = BTLElectronicsToDetIdMappingSoALayout<>;

#endif
