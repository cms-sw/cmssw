#ifndef DataFormats_HcalDetId_HcalZDCDetId_h_included
#define DataFormats_HcalDetId_HcalZDCDetId_h_included 1

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

/** \class HcalZDCDetId
  *  
  *  Contents of the HcalZDCDetId (Old):
  *     [7]   Set for RPD
  *     [6]   Z position (true for positive)
  *     [5:4] Section (EM/HAD/Lumi)
  *     [3:0] Channel
  *  Contents of the HcalZDCDetId (New):
  *     [8]   Set to 1 for new format
  *     [7]   Set for RPD
  *     [6]   Z position (true for positive)
  *     [5:4] Section (EM/HAD/Lumi)
  *     [3:0] Chaneel for EM/HAD/Lumi
  *     [5:0] Channel for RPD
  *
  */
class HcalZDCDetId : public DetId {
public:
  static constexpr uint32_t kZDCChannelMask1 = 0xF;
  static constexpr uint32_t kZDCChannelMask2 = 0x7F;
  static constexpr uint32_t kZDCSectionMask = 0x3;
  static constexpr uint32_t kZDCSectionOffset = 4;
  static constexpr uint32_t kZDCZsideMask = 0x40;
  static constexpr uint32_t kZDCRPDMask = 0x80;
  static constexpr uint32_t kZDCnewFormat = 0x100;
  enum Section { Unknown = 0, EM = 1, HAD = 2, LUM = 3, RPD = 4 };

  static constexpr int32_t SubdetectorId = 2;

  static constexpr int32_t kDepEM = 5;
  static constexpr int32_t kDepHAD = 4;
  static constexpr int32_t kDepLUM = 2;
  static constexpr int32_t kDepRPD = 16;
  static constexpr int32_t kDepRun1 = (kDepEM + kDepHAD + kDepLUM);
  static constexpr int32_t kDepTot = (kDepRun1 + kDepRPD);
  static constexpr int32_t kDepRun3 = kDepTot;

  /** Create a null cellid*/
  constexpr HcalZDCDetId() : DetId() {}
  /** Create cellid from raw id (0=invalid tower id) */
  constexpr HcalZDCDetId(uint32_t rawid) : DetId(rawid) {}
  /** Constructor from section, eta sign, and channel */
  constexpr HcalZDCDetId(Section section, bool true_for_positive_eta, int32_t channel) {
    id_ = packHcalZDCDetId(section, true_for_positive_eta, channel);
  }
  /** Constructor from a generic cell id */
  constexpr HcalZDCDetId(const DetId& gen) {
    if (!gen.null() && (gen.det() != Calo || gen.subdetId() != SubdetectorId)) {
      throw cms::Exception("Invalid DetId")
          << "Cannot initialize ZDCDetId from " << std::hex << gen.rawId() << std::dec;
    }
    id_ = newForm(gen.rawId());
  }
  /** Assignment from a generic cell id */
  constexpr HcalZDCDetId& operator=(const DetId& gen) {
    if (!gen.null() && (gen.det() != Calo || gen.subdetId() != SubdetectorId)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign ZDCDetId from " << std::hex << gen.rawId() << std::dec;
    }
    id_ = newForm(gen.rawId());
    return *this;
  }
  /** Comparison operator */
  constexpr bool operator==(DetId gen) const {
    if (gen.rawId() == id_) {
      return true;
    } else {
      uint32_t id1 = newForm(gen.rawId());
      uint32_t id2 = newForm(id_);
      return (id1 == id2);
    }
  }
  constexpr bool operator!=(DetId gen) const {
    if (gen.rawId() != id_) {
      return true;
    } else {
      uint32_t id1 = newForm(gen.rawId());
      uint32_t id2 = newForm(id_);
      return (id1 != id2);
    }
  }

  /// get the z-side of the cell (1/-1)
  constexpr int32_t zside() const { return ((id_ & kZDCZsideMask) ? (1) : (-1)); }
  /// get the section
  constexpr Section section() const {
    uint32_t id = newForm(id_);
    if (id & kZDCRPDMask)
      return RPD;
    else
      return (Section)((id >> kZDCSectionOffset) & kZDCSectionMask);
  }
  /// get the depth (1 for EM, channel + 1 for HAD, 2 for RPD, not sure yet for LUM, leave as default)
  constexpr int32_t depth() const {
    const int se(section());
    if (se == EM)
      return 1;
    else if (se == HAD)
      return (channel() + 2);
    else if (se == RPD)
      return 2;
    else
      return channel();
  }
  /// get the channel
  constexpr int32_t channel() const {
    const int32_t se(section());
    uint32_t id = newForm(id_);
    if (se == RPD)
      return (1 + (id & kZDCChannelMask2));
    else
      return (id & kZDCChannelMask1);
  }

  constexpr static bool newFormat(const uint32_t& di) { return (di & kZDCnewFormat); }
  constexpr static uint32_t newForm(const uint32_t& di) {
    uint32_t id(di);
    if (!newFormat(id)) {
      Section se(Unknown);
      bool zside(true);
      int32_t channel(0);
      unpackHcalZDCDetId(id, se, zside, channel);
      id = packHcalZDCDetId(se, zside, channel);
    }
    return id;
  }

  constexpr uint32_t denseIndex() const {
    const int32_t se(section());
    uint32_t di =
        (channel() - 1 +
         (se == RPD ? 2 * kDepRun1 + (zside() < 0 ? 0 : kDepRPD)
                    : ((zside() < 0 ? 0 : kDepRun1) + (se == HAD ? kDepEM : (se == LUM ? kDepEM + kDepHAD : 0)))));
    return di;
  }

  constexpr static bool validDenseIndex(const uint32_t& di) { return (di < kSizeForDenseIndexing); }

  constexpr static HcalZDCDetId detIdFromDenseIndex(uint32_t di) {
    if (validDenseIndex(di)) {
      bool lz(false);
      uint32_t dp(0);
      Section se(Unknown);
      if (di >= 2 * kDepRun1) {
        lz = (di >= (kDepRun1 + kDepTot));
        se = RPD;
        dp = 1 + ((di - 2 * kDepRun1) % kDepRPD);
      } else {
        lz = (di >= kDepRun1);
        uint32_t in = (di % kDepRun1);
        se = (in < kDepEM ? EM : (in < kDepEM + kDepHAD ? HAD : LUM));
        dp = (se == EM ? in + 1 : (se == HAD ? in - kDepEM + 1 : in - kDepEM - kDepHAD + 1));
      }
      return HcalZDCDetId(se, lz, dp);
    } else {
      return HcalZDCDetId();
    }
  }

  constexpr static bool validDetId(Section se, int32_t dp) {
    bool flag = (dp >= 1 && (((se == EM) && (dp <= kDepEM)) || ((se == HAD) && (dp <= kDepHAD)) ||
                             ((se == LUM) && (dp <= kDepLUM)) || ((se == RPD) && (dp <= kDepRPD))));
    return flag;
  }

private:
  constexpr static uint32_t packHcalZDCDetId(const Section& se, const bool& zside, const int32_t& channel) {
    uint32_t id = DetId(DetId::Calo, SubdetectorId);
    id |= kZDCnewFormat;
    if (se == RPD) {
      id |= kZDCRPDMask;
      id |= ((channel - 1) & kZDCChannelMask2);
    } else {
      id |= (se & kZDCSectionMask) << kZDCSectionOffset;
      id |= (channel & kZDCChannelMask1);
    }
    if (zside)
      id |= kZDCZsideMask;
    return id;
  }

  constexpr static void unpackHcalZDCDetId(const uint32_t& id, Section& se, bool& zside, int32_t& channel) {
    if (id & kZDCnewFormat) {
      se = (id & kZDCRPDMask) ? RPD : (Section)((id >> kZDCSectionOffset) & kZDCSectionMask);
      channel = (se == RPD) ? (1 + (id & kZDCChannelMask2)) : (id & kZDCChannelMask1);
      zside = (id & kZDCZsideMask);
    } else {
      se = (id & kZDCRPDMask) ? RPD : (Section)((id >> kZDCSectionOffset) & kZDCSectionMask);
      channel = (se == RPD) ? (1 + (id & kZDCChannelMask1)) : (id & kZDCChannelMask1);
      zside = (id & kZDCZsideMask);
    }
  }

public:
  constexpr static int32_t kSizeForDenseIndexingRun1 = 2 * kDepRun1;
  constexpr static int32_t kSizeForDenseIndexingRun3 = 2 * kDepRun3;
  enum { kSizeForDenseIndexing = kSizeForDenseIndexingRun1 };
};

std::ostream& operator<<(std::ostream&, const HcalZDCDetId& id);

#endif  // DataFormats_HcalDetId_HcalZDCDetId_h_included
