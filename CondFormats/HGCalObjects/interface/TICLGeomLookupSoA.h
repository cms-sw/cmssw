#ifndef CondFormats_HGCalObjects_interface_TICLGeomLookupSoA_h
#define CondFormats_HGCalObjects_interface_TICLGeomLookupSoA_h

#include <cstdint>

#include "CondFormats/HGCalObjects/interface/TICLGeomSoA.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Coarse-block lookup table: maps a detid to its dense id (row in the cells
// SoA) by a bit-shift coarse key and a short in-block binary search. The
// cells SoA is ordered by rawDetId, which groups every cell of one silicon wafer
// (or one scintillator ring) into a contiguous block, because the cell
// coordinates occupy the low bits of the detid (silicon: cellU/cellV in
// bits 0..9; scintillator: iphi in bits 0..8). The coarse index of a detid
// is therefore a pure bit shift + mask, and this table maps that index to
// the block's start row and length in the cells SoA. denseIdOf then does a
// short binary search inside the block. One table per HGCal subdetector
// (EE, HSi, HSc), concatenated; the per-subdetector base offsets are
// scalars. Works identically on host and device.
GENERATE_SOA_LAYOUT(TICLGeomLookupSoALayout,
                    SOA_COLUMN(int32_t, blockStart),
                    SOA_COLUMN(int32_t, blockCount),
                    SOA_SCALAR(int32_t, eeBase),
                    SOA_SCALAR(int32_t, hsiBase),
                    SOA_SCALAR(int32_t, hscBase))

using TICLGeomLookupSoA = TICLGeomLookupSoALayout<>;
using TICLGeomLookupSoAView = TICLGeomLookupSoA::View;
using TICLGeomLookupSoAConstView = TICLGeomLookupSoA::ConstView;

namespace ticlgeom {

  // Detector enum values (DetId::HGCalEE, HGCalHSi, HGCalHSc) and detid
  // field layout, mirrored here so the coarse index is computable from raw
  // bits without pulling the DetId headers into device code.
  namespace detail {
    constexpr uint32_t kDetOffset = 28, kDetMask = 0xF;
    constexpr uint32_t kHGCalEE = 8, kHGCalHSi = 9, kHGCalHSc = 10;
    // silicon: cellU/cellV occupy bits 0..9, so the wafer key is bits 10..27
    constexpr uint32_t kSiCellBits = 10, kSiKeyMask = 0x3FFFF;
    // scintillator: iphi occupies bits 0..8, so the ring key is bits 9..27
    constexpr uint32_t kScCellBits = 9, kScKeyMask = 0x7FFFF;
    constexpr int32_t kSiSlots = kSiKeyMask + 1;  // 262144
    constexpr int32_t kScSlots = kScKeyMask + 1;  // 524288
  }  // namespace detail

  // Slot in the concatenated block table for a detid, or -1 for a detid of
  // no supported subdetector.
  SOA_HOST_DEVICE SOA_INLINE int32_t coarseSlot(TICLGeomLookupSoAConstView const& lookup, uint32_t rawDetId) {
    using namespace detail;
    const uint32_t det = (rawDetId >> kDetOffset) & kDetMask;
    if (det == kHGCalEE) {
      return lookup.eeBase() + static_cast<int32_t>((rawDetId >> kSiCellBits) & kSiKeyMask);
    } else if (det == kHGCalHSi) {
      return lookup.hsiBase() + static_cast<int32_t>((rawDetId >> kSiCellBits) & kSiKeyMask);
    } else if (det == kHGCalHSc) {
      return lookup.hscBase() + static_cast<int32_t>((rawDetId >> kScCellBits) & kScKeyMask);
    }
    return -1;
  }

  // O(1) coarse index + short binary search within the block. Returns the
  // dense id (row in the cells SoA) of rawDetId, or -1 if absent.
  SOA_HOST_DEVICE SOA_INLINE int32_t denseIdOf(TICLGeomLookupSoAConstView const& lookup,
                                               TICLGeomCommonSoAConstView const& common,
                                               uint32_t rawDetId) {
    const int32_t slot = coarseSlot(lookup, rawDetId);
    if (slot < 0) {
      // non-HGCal detid (barrel cells in an ECAL/HCAL/withBarrel instance):
      // the arithmetic coarse index does not cover it, fall back to a full
      // binary search over the sorted cells so lookups stay correct
      return indexOf(common, rawDetId);
    }
    const int32_t start = lookup[slot].blockStart();
    const int32_t count = lookup[slot].blockCount();
    if (count <= 0) {
      return -1;
    }
    int32_t lo = start, hi = start + count - 1;
    while (lo <= hi) {
      const int32_t mid = lo + (hi - lo) / 2;
      const uint32_t val = common[mid].rawDetId();
      if (val == rawDetId) {
        return mid;
      }
      if (val < rawDetId) {
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return -1;
  }

}  // namespace ticlgeom

#endif  // CondFormats_HGCalObjects_interface_TICLGeomLookupSoA_h
