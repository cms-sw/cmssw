#ifndef CondFormats_Phase2TrackerDTC_DTCELinkId_h
#define CondFormats_Phase2TrackerDTC_DTCELinkId_h

// -*- C++ -*-
//
// Package:    CondFormats/Phase2TrackerDTC
// Class:      DTCELinkId
//
/**\class DTCELinkId DTCELinkId.cc CondFormats/Phase2TrackerDTC/src/DTCELinkId.cc

Description: DTCELinkId identifies a specific eLink in the interface of a specific GBT link instance in the firmware of a specific DTC of the tracker back-end.

Implementation:
		[Notes on implementation]
*/
//
// Original Author:  Luigi Calligaris, SPRACE, Sao Paulo, BR
// Created        :  Wed, 27 Feb 2019 21:41:13 GMT
//
//

#include "CondFormats/Serialization/interface/Serializable.h"

#include <cstdint>
#include <functional>
#include <limits>

class DTCELinkId {
public:
  DTCELinkId() noexcept;
  DTCELinkId(DTCELinkId const&) noexcept;
  DTCELinkId(DTCELinkId&&) noexcept;
  DTCELinkId& operator=(DTCELinkId const&) noexcept;
  DTCELinkId& operator=(DTCELinkId&&) noexcept;
  ~DTCELinkId() noexcept;

  // Constructs a DTCELinkId addressed by (dtc_id, gbtlink_id, elink_id)
  DTCELinkId(uint16_t, uint8_t, uint8_t) noexcept;

  inline auto elink_id() const noexcept { return elink_id_; }
  inline auto gbtlink_id() const noexcept { return gbtlink_id_; }
  inline auto dtc_id() const noexcept { return dtc_id_; }

private:
  // In order to keep the payload small, we use the C standard integers, optimizing them for size.
  // The lpGBT has at most 7 ePorts, therefore they can be addressed by an 8-bit number.
  // The DTC should host at most 72 GBT links, therefore an 8-bit number should be enough to address it.
  // The C++ memory alignment and padding rules impose that this class will have at least 32 bits size,
  // i.e. 8+8+8 bits and 8+8+16 would be the same, so we choose the latter.
  uint8_t elink_id_;
  uint8_t gbtlink_id_;
  uint16_t dtc_id_;

  COND_SERIALIZABLE;
};

namespace std {
  template <>
  struct hash<DTCELinkId> {
    size_t operator()(const DTCELinkId& k) const noexcept {
      // With
      constexpr const size_t shift_gbtlink_id = numeric_limits<decltype(k.elink_id())>::max() + 1u;
      constexpr const size_t shift_dtc_id = (numeric_limits<decltype(k.gbtlink_id())>::max() + 1u) * shift_gbtlink_id;

      return k.elink_id() + k.gbtlink_id() * shift_gbtlink_id + k.dtc_id() * shift_dtc_id;
    }
  };
}  // namespace std

inline bool operator<(DTCELinkId const& lhs, DTCELinkId const& rhs) {
  return lhs.dtc_id() < rhs.dtc_id() || (lhs.dtc_id() == rhs.dtc_id() && lhs.gbtlink_id() < rhs.gbtlink_id()) ||
         (lhs.dtc_id() == rhs.dtc_id() && lhs.gbtlink_id() == rhs.gbtlink_id() && lhs.elink_id() < rhs.elink_id());
}
inline bool operator>(DTCELinkId const& lhs, DTCELinkId const& rhs) {
  return lhs.dtc_id() > rhs.dtc_id() || (lhs.dtc_id() == rhs.dtc_id() && lhs.gbtlink_id() > rhs.gbtlink_id()) ||
         (lhs.dtc_id() == rhs.dtc_id() && lhs.gbtlink_id() == rhs.gbtlink_id() && lhs.elink_id() > rhs.elink_id());
}
inline bool operator==(DTCELinkId const& lhs, DTCELinkId const& rhs) {
  return lhs.dtc_id() == rhs.dtc_id() && lhs.gbtlink_id() == rhs.gbtlink_id() && lhs.elink_id() == rhs.elink_id();
}
inline bool operator!=(DTCELinkId const& lhs, DTCELinkId const& rhs) {
  return lhs.dtc_id() != rhs.dtc_id() || lhs.gbtlink_id() != rhs.gbtlink_id() || lhs.elink_id() != rhs.elink_id();
}

#endif  // end DataFormats_Phase2TrackerDTC_DTCELinkId_h
