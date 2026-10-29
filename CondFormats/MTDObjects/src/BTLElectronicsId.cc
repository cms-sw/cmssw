#include "CondFormats/MTDObjects/interface/BTLElectronicsId.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsSpecs.h"

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

uint32_t BTLElectronicsId::denseChannelIndex() const {
  using namespace BTLElectronicsSpecs;
  const uint32_t serenityIdx = (fedId() - kFirstFEDId) / nFEDsPerSerenity;  // 0-11
  return ((serenityIdx * nHSLinksPerSerenity + hsLinkId()) * nELinksPerHSLink + eLinkId()) * nChannelsPerELink +
         channelId();
}

bool BTLElectronicsId::isValid() const {
  using namespace BTLElectronicsSpecs;
  const int fed = fedId() - static_cast<int>(kFirstFEDId);
  const int hs = hsLinkId();
  // FED k of a Serenity carries the HS-links [k * nHSLinksPerFED, (k + 1) * nHSLinksPerFED)
  return fed >= 0 && fed < static_cast<int>(kNumberOfFEDs) && hs < nHSLinksPerSerenity &&
         hs / nHSLinksPerFED == fed % nFEDsPerSerenity && eLinkId() < nELinksPerHSLink &&
         channelId() < nChannelsPerELink;
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
     << ", e-link = " << id.eLinkId() << ", TOFHIR channel Id = " << id.channelId()
     << ", dense channel index = " << id.denseChannelIndex();

  return os;
}
