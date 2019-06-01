#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"

#include <string>
#include <functional>
#include <limits>

DTCELinkId::DTCELinkId() noexcept
    : elink_id_(std::numeric_limits<decltype(elink_id())>::max()),
      gbtlink_id_(std::numeric_limits<decltype(gbtlink_id())>::max()),
      dtc_id_(std::numeric_limits<decltype(dtc_id())>::max()) {}

DTCELinkId::DTCELinkId(DTCELinkId const& rhs) noexcept
    : elink_id_(rhs.elink_id_), gbtlink_id_(rhs.gbtlink_id_), dtc_id_(rhs.dtc_id_) {}

DTCELinkId::DTCELinkId(DTCELinkId&& rhs) noexcept
    : elink_id_(rhs.elink_id_), gbtlink_id_(rhs.gbtlink_id_), dtc_id_(rhs.dtc_id_) {}

DTCELinkId& DTCELinkId::operator=(DTCELinkId const& rhs) noexcept {
  elink_id_ = rhs.elink_id_;
  gbtlink_id_ = rhs.gbtlink_id_;
  dtc_id_ = rhs.dtc_id_;

  return *this;
}

DTCELinkId& DTCELinkId::operator=(DTCELinkId&& rhs) noexcept {
  elink_id_ = rhs.elink_id_;
  gbtlink_id_ = rhs.gbtlink_id_;
  dtc_id_ = rhs.dtc_id_;

  return *this;
}

DTCELinkId::~DTCELinkId() noexcept {}

DTCELinkId::DTCELinkId(uint16_t dtc_id, uint8_t gbtlink_id, uint8_t elink_id) noexcept
    : elink_id_(elink_id), gbtlink_id_(gbtlink_id), dtc_id_(dtc_id) {}
