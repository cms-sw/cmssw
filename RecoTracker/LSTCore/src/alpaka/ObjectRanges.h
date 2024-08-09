#ifndef RecoTracker_LSTCore_interface_ObjectRanges_h
#define RecoTracker_LSTCore_interface_ObjectRanges_h

#include "RecoTracker/LSTCore/interface/Constants.h"

namespace lst {

  struct ObjectRanges {
    int* hitRanges;
    int* hitRangesLower;
    int* hitRangesUpper;
    int8_t* hitRangesnLower;
    int8_t* hitRangesnUpper;
    int* mdRanges;
    int* segmentRanges;
    int* trackletRanges;
    int* tripletRanges;
    int* trackCandidateRanges;
    // Others will be added later
    int* quintupletRanges;

    // This number is just nEligibleModules - 1, but still we want this to be independent of the TC kernel
    uint16_t* nEligibleT5Modules;
    // Will be allocated in createQuintuplets kernel!
    uint16_t* indicesOfEligibleT5Modules;
    // To store different starting points for variable occupancy stuff
    int* quintupletModuleIndices;
    int* quintupletModuleOccupancy;
    int* miniDoubletModuleIndices;
    int* miniDoubletModuleOccupancy;
    int* segmentModuleIndices;
    int* segmentModuleOccupancy;
    int* tripletModuleIndices;
    int* tripletModuleOccupancy;

    unsigned int* device_nTotalMDs;
    unsigned int* device_nTotalSegs;
    unsigned int* device_nTotalTrips;
    unsigned int* device_nTotalQuints;

    template <typename TBuff>
    void setData(TBuff& buf) {
      hitRanges = alpaka::getPtrNative(buf.hitRanges_buf);
      hitRangesLower = alpaka::getPtrNative(buf.hitRangesLower_buf);
      hitRangesUpper = alpaka::getPtrNative(buf.hitRangesUpper_buf);
      hitRangesnLower = alpaka::getPtrNative(buf.hitRangesnLower_buf);
      hitRangesnUpper = alpaka::getPtrNative(buf.hitRangesnUpper_buf);
      mdRanges = alpaka::getPtrNative(buf.mdRanges_buf);
      segmentRanges = alpaka::getPtrNative(buf.segmentRanges_buf);
      trackletRanges = alpaka::getPtrNative(buf.trackletRanges_buf);
      tripletRanges = alpaka::getPtrNative(buf.tripletRanges_buf);
      trackCandidateRanges = alpaka::getPtrNative(buf.trackCandidateRanges_buf);
      quintupletRanges = alpaka::getPtrNative(buf.quintupletRanges_buf);

      nEligibleT5Modules = alpaka::getPtrNative(buf.nEligibleT5Modules_buf);
      indicesOfEligibleT5Modules = alpaka::getPtrNative(buf.indicesOfEligibleT5Modules_buf);

      quintupletModuleIndices = alpaka::getPtrNative(buf.quintupletModuleIndices_buf);
      quintupletModuleOccupancy = alpaka::getPtrNative(buf.quintupletModuleOccupancy_buf);
      miniDoubletModuleIndices = alpaka::getPtrNative(buf.miniDoubletModuleIndices_buf);
      miniDoubletModuleOccupancy = alpaka::getPtrNative(buf.miniDoubletModuleOccupancy_buf);
      segmentModuleIndices = alpaka::getPtrNative(buf.segmentModuleIndices_buf);
      segmentModuleOccupancy = alpaka::getPtrNative(buf.segmentModuleOccupancy_buf);
      tripletModuleIndices = alpaka::getPtrNative(buf.tripletModuleIndices_buf);
      tripletModuleOccupancy = alpaka::getPtrNative(buf.tripletModuleOccupancy_buf);

      device_nTotalMDs = alpaka::getPtrNative(buf.device_nTotalMDs_buf);
      device_nTotalSegs = alpaka::getPtrNative(buf.device_nTotalSegs_buf);
      device_nTotalTrips = alpaka::getPtrNative(buf.device_nTotalTrips_buf);
      device_nTotalQuints = alpaka::getPtrNative(buf.device_nTotalQuints_buf);
    }
  };

  template <typename TDev>
  struct ObjectRangesBuffer {
    Buf<TDev, int> hitRanges_buf;
    Buf<TDev, int> hitRangesLower_buf;
    Buf<TDev, int> hitRangesUpper_buf;
    Buf<TDev, int8_t> hitRangesnLower_buf;
    Buf<TDev, int8_t> hitRangesnUpper_buf;
    Buf<TDev, int> mdRanges_buf;
    Buf<TDev, int> segmentRanges_buf;
    Buf<TDev, int> trackletRanges_buf;
    Buf<TDev, int> tripletRanges_buf;
    Buf<TDev, int> trackCandidateRanges_buf;
    Buf<TDev, int> quintupletRanges_buf;

    Buf<TDev, uint16_t> nEligibleT5Modules_buf;
    Buf<TDev, uint16_t> indicesOfEligibleT5Modules_buf;

    Buf<TDev, int> quintupletModuleIndices_buf;
    Buf<TDev, int> quintupletModuleOccupancy_buf;
    Buf<TDev, int> miniDoubletModuleIndices_buf;
    Buf<TDev, int> miniDoubletModuleOccupancy_buf;
    Buf<TDev, int> segmentModuleIndices_buf;
    Buf<TDev, int> segmentModuleOccupancy_buf;
    Buf<TDev, int> tripletModuleIndices_buf;
    Buf<TDev, int> tripletModuleOccupancy_buf;

    Buf<TDev, unsigned int> device_nTotalMDs_buf;
    Buf<TDev, unsigned int> device_nTotalSegs_buf;
    Buf<TDev, unsigned int> device_nTotalTrips_buf;
    Buf<TDev, unsigned int> device_nTotalQuints_buf;

    ObjectRanges data_;

    template <typename TQueue, typename TDevAcc>
    ObjectRangesBuffer(unsigned int nMod, unsigned int nLowerMod, TDevAcc const& devAccIn, TQueue& queue)
        : hitRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          hitRangesLower_buf(allocBufWrapper<int>(devAccIn, nMod, queue)),
          hitRangesUpper_buf(allocBufWrapper<int>(devAccIn, nMod, queue)),
          hitRangesnLower_buf(allocBufWrapper<int8_t>(devAccIn, nMod, queue)),
          hitRangesnUpper_buf(allocBufWrapper<int8_t>(devAccIn, nMod, queue)),
          mdRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          segmentRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          trackletRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          tripletRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          trackCandidateRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          quintupletRanges_buf(allocBufWrapper<int>(devAccIn, nMod * 2, queue)),
          nEligibleT5Modules_buf(allocBufWrapper<uint16_t>(devAccIn, 1, queue)),
          indicesOfEligibleT5Modules_buf(allocBufWrapper<uint16_t>(devAccIn, nLowerMod, queue)),
          quintupletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod, queue)),
          quintupletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod, queue)),
          miniDoubletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod + 1, queue)),
          miniDoubletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod + 1, queue)),
          segmentModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod + 1, queue)),
          segmentModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod + 1, queue)),
          tripletModuleIndices_buf(allocBufWrapper<int>(devAccIn, nLowerMod, queue)),
          tripletModuleOccupancy_buf(allocBufWrapper<int>(devAccIn, nLowerMod, queue)),
          device_nTotalMDs_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          device_nTotalSegs_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          device_nTotalTrips_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)),
          device_nTotalQuints_buf(allocBufWrapper<unsigned int>(devAccIn, 1, queue)) {
      alpaka::memset(queue, hitRanges_buf, 0xff);
      alpaka::memset(queue, hitRangesLower_buf, 0xff);
      alpaka::memset(queue, hitRangesUpper_buf, 0xff);
      alpaka::memset(queue, hitRangesnLower_buf, 0xff);
      alpaka::memset(queue, hitRangesnUpper_buf, 0xff);
      alpaka::memset(queue, mdRanges_buf, 0xff);
      alpaka::memset(queue, segmentRanges_buf, 0xff);
      alpaka::memset(queue, trackletRanges_buf, 0xff);
      alpaka::memset(queue, tripletRanges_buf, 0xff);
      alpaka::memset(queue, trackCandidateRanges_buf, 0xff);
      alpaka::memset(queue, quintupletRanges_buf, 0xff);
      alpaka::memset(queue, quintupletModuleIndices_buf, 0xff);
      alpaka::wait(queue);
      data_.setData(*this);
    }

    inline ObjectRanges const* data() const { return &data_; }
    void setData(ObjectRangesBuffer& buf) { data_.setData(buf); }
  };

}  // namespace lst
#endif
