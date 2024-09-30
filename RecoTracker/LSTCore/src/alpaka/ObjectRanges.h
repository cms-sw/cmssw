#ifndef RecoTracker_LSTCore_interface_ObjectRanges_h
#define RecoTracker_LSTCore_interface_ObjectRanges_h

#include "RecoTracker/LSTCore/interface/Constants.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

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
      hitRanges = buf.hitRanges_buf.data();
      hitRangesLower = buf.hitRangesLower_buf.data();
      hitRangesUpper = buf.hitRangesUpper_buf.data();
      hitRangesnLower = buf.hitRangesnLower_buf.data();
      hitRangesnUpper = buf.hitRangesnUpper_buf.data();
      mdRanges = buf.mdRanges_buf.data();
      segmentRanges = buf.segmentRanges_buf.data();
      trackletRanges = buf.trackletRanges_buf.data();
      tripletRanges = buf.tripletRanges_buf.data();
      trackCandidateRanges = buf.trackCandidateRanges_buf.data();
      quintupletRanges = buf.quintupletRanges_buf.data();

      nEligibleT5Modules = buf.nEligibleT5Modules_buf.data();
      indicesOfEligibleT5Modules = buf.indicesOfEligibleT5Modules_buf.data();

      quintupletModuleIndices = buf.quintupletModuleIndices_buf.data();
      quintupletModuleOccupancy = buf.quintupletModuleOccupancy_buf.data();
      miniDoubletModuleIndices = buf.miniDoubletModuleIndices_buf.data();
      miniDoubletModuleOccupancy = buf.miniDoubletModuleOccupancy_buf.data();
      segmentModuleIndices = buf.segmentModuleIndices_buf.data();
      segmentModuleOccupancy = buf.segmentModuleOccupancy_buf.data();
      tripletModuleIndices = buf.tripletModuleIndices_buf.data();
      tripletModuleOccupancy = buf.tripletModuleOccupancy_buf.data();

      device_nTotalMDs = buf.device_nTotalMDs_buf.data();
      device_nTotalSegs = buf.device_nTotalSegs_buf.data();
      device_nTotalTrips = buf.device_nTotalTrips_buf.data();
      device_nTotalQuints = buf.device_nTotalQuints_buf.data();
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
      data_.setData(*this);
    }

    inline ObjectRanges const* data() const { return &data_; }
    void setData(ObjectRangesBuffer& buf) { data_.setData(buf); }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
