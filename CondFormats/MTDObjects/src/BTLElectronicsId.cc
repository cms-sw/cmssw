#include "CondFormats/MTDObjects/interface/BTLElectronicsId.h"

#include <ostream>

// ------------------------------------------------------------
// Constructors
// ------------------------------------------------------------
BTLElectronicsId::BTLElectronicsId() : rawid_(0) {}

BTLElectronicsId::BTLElectronicsId(uint32_t rawid) : rawid_(rawid) {}

BTLElectronicsId::BTLElectronicsId(int fed, int hsLink, int eLink, int channel) {
  rawid_ = (static_cast<uint32_t>(fed) << kFEDShift) | (static_cast<uint32_t>(hsLink) << kHSLinkShift) |
           (static_cast<uint32_t>(eLink) << kELinkShift) | (static_cast<uint32_t>(channel) << kChannelShift);
}

// ------------------------------------------------------------
// Comparison operators
// ------------------------------------------------------------

bool BTLElectronicsId::operator==(const BTLElectronicsId& other) const { return rawid_ == other.rawid_; }

bool BTLElectronicsId::operator!=(const BTLElectronicsId& other) const { return rawid_ != other.rawid_; }

// ------------------------------------------------------------
// Stream operator
// ------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, const BTLElectronicsId& id) {
  os << "BTLElectronicsId: "
     << "rawId = " << static_cast<int>(id.rawId()) << ", FED Id = " << id.fedId() << ", HS-link  = " << id.hsLinkId()
     << ", e-link = " << id.eLinkId() << ", TOFHIR channel Id = " << id.channelId();

  return os;
}
