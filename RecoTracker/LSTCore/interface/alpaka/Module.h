#ifndef Module_cuh
#define Module_cuh

#include <alpaka/alpaka.hpp>

#ifdef LST_IS_CMSSW_PACKAGE
#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#else
#include "Constants.h"
#endif

namespace SDL {
  enum SubDet { InnerPixel = 0, Barrel = 5, Endcap = 4 };

  enum Side { NegZ = 1, PosZ = 2, Center = 3 };

  enum ModuleType { PS, TwoS, PixelModule };

  enum ModuleLayerType { Pixel, Strip, InnerPixelLayer };

  struct objectRanges {
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
    void setData(TBuff& objectRangesbuf) {
      hitRanges = alpaka::getPtrNative(objectRangesbuf.hitRanges_buf);
      hitRangesLower = alpaka::getPtrNative(objectRangesbuf.hitRangesLower_buf);
      hitRangesUpper = alpaka::getPtrNative(objectRangesbuf.hitRangesUpper_buf);
      hitRangesnLower = alpaka::getPtrNative(objectRangesbuf.hitRangesnLower_buf);
      hitRangesnUpper = alpaka::getPtrNative(objectRangesbuf.hitRangesnUpper_buf);
      mdRanges = alpaka::getPtrNative(objectRangesbuf.mdRanges_buf);
      segmentRanges = alpaka::getPtrNative(objectRangesbuf.segmentRanges_buf);
      trackletRanges = alpaka::getPtrNative(objectRangesbuf.trackletRanges_buf);
      tripletRanges = alpaka::getPtrNative(objectRangesbuf.tripletRanges_buf);
      trackCandidateRanges = alpaka::getPtrNative(objectRangesbuf.trackCandidateRanges_buf);
      quintupletRanges = alpaka::getPtrNative(objectRangesbuf.quintupletRanges_buf);

      nEligibleT5Modules = alpaka::getPtrNative(objectRangesbuf.nEligibleT5Modules_buf);
      indicesOfEligibleT5Modules = alpaka::getPtrNative(objectRangesbuf.indicesOfEligibleT5Modules_buf);

      quintupletModuleIndices = alpaka::getPtrNative(objectRangesbuf.quintupletModuleIndices_buf);
      quintupletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.quintupletModuleOccupancy_buf);
      miniDoubletModuleIndices = alpaka::getPtrNative(objectRangesbuf.miniDoubletModuleIndices_buf);
      miniDoubletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.miniDoubletModuleOccupancy_buf);
      segmentModuleIndices = alpaka::getPtrNative(objectRangesbuf.segmentModuleIndices_buf);
      segmentModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.segmentModuleOccupancy_buf);
      tripletModuleIndices = alpaka::getPtrNative(objectRangesbuf.tripletModuleIndices_buf);
      tripletModuleOccupancy = alpaka::getPtrNative(objectRangesbuf.tripletModuleOccupancy_buf);

      device_nTotalMDs = alpaka::getPtrNative(objectRangesbuf.device_nTotalMDs_buf);
      device_nTotalSegs = alpaka::getPtrNative(objectRangesbuf.device_nTotalSegs_buf);
      device_nTotalTrips = alpaka::getPtrNative(objectRangesbuf.device_nTotalTrips_buf);
      device_nTotalQuints = alpaka::getPtrNative(objectRangesbuf.device_nTotalQuints_buf);
    }
  };

  template <typename TDev>
  struct objectRangesBuffer : objectRanges {
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

    template <typename TQueue, typename TDevAcc>
    objectRangesBuffer(unsigned int nMod, unsigned int nLowerMod, TDevAcc const& devAccIn, TQueue& queue)
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
    }
  };

  struct modules {
    const unsigned int* detIds;
    const uint16_t* moduleMap;
    const unsigned int* mapdetId;
    const uint16_t* mapIdx;
    const uint16_t* nConnectedModules;
    const float* drdzs;
    const float* dxdys;
    const uint16_t* nModules;
    const uint16_t* nLowerModules;
    const uint16_t* partnerModuleIndices;

    const short* layers;
    const short* rings;
    const short* modules;
    const short* rods;
    const short* subdets;
    const short* sides;
    const float* eta;
    const float* r;
    const bool* isInverted;
    const bool* isLower;
    const bool* isAnchor;
    const ModuleType* moduleType;
    const ModuleLayerType* moduleLayerType;
    const int* sdlLayers;
    const unsigned int* connectedPixels;

    static bool parseIsInverted(short subdet, short side, short module, short layer) {
      if (subdet == Endcap) {
        if (side == NegZ) {
          return module % 2 == 1;
        } else if (side == PosZ) {
          return module % 2 == 0;
        } else {
          return false;
        }
      } else if (subdet == Barrel) {
        if (side == Center) {
          if (layer <= 3) {
            return module % 2 == 1;
          } else if (layer >= 4) {
            return module % 2 == 0;
          } else {
            return false;
          }
        } else if (side == NegZ or side == PosZ) {
          if (layer <= 2) {
            return module % 2 == 1;
          } else if (layer == 3) {
            return module % 2 == 0;
          } else {
            return false;
          }
        } else {
          return false;
        }
      } else {
        return false;
      }
    };

    static bool parseIsLower(bool isInvertedx, unsigned int detId) {
      return (isInvertedx) ? !(detId & 1) : (detId & 1);
    };

    static unsigned int parsePartnerModuleId(unsigned int detId, bool isLowerx, bool isInvertedx) {
      return isLowerx ? (isInvertedx ? detId - 1 : detId + 1) : (isInvertedx ? detId + 1 : detId - 1);
    };

    template <typename TBuff>
    void setData(const TBuff& modulesbuf) {
      detIds = alpaka::getPtrNative(modulesbuf.detIds_buf);
      moduleMap = alpaka::getPtrNative(modulesbuf.moduleMap_buf);
      mapdetId = alpaka::getPtrNative(modulesbuf.mapdetId_buf);
      mapIdx = alpaka::getPtrNative(modulesbuf.mapIdx_buf);
      nConnectedModules = alpaka::getPtrNative(modulesbuf.nConnectedModules_buf);
      drdzs = alpaka::getPtrNative(modulesbuf.drdzs_buf);
      dxdys = alpaka::getPtrNative(modulesbuf.dxdys_buf);
      nModules = alpaka::getPtrNative(modulesbuf.nModules_buf);
      nLowerModules = alpaka::getPtrNative(modulesbuf.nLowerModules_buf);
      partnerModuleIndices = alpaka::getPtrNative(modulesbuf.partnerModuleIndices_buf);

      layers = alpaka::getPtrNative(modulesbuf.layers_buf);
      rings = alpaka::getPtrNative(modulesbuf.rings_buf);
      modules = alpaka::getPtrNative(modulesbuf.modules_buf);
      rods = alpaka::getPtrNative(modulesbuf.rods_buf);
      subdets = alpaka::getPtrNative(modulesbuf.subdets_buf);
      sides = alpaka::getPtrNative(modulesbuf.sides_buf);
      eta = alpaka::getPtrNative(modulesbuf.eta_buf);
      r = alpaka::getPtrNative(modulesbuf.r_buf);
      isInverted = alpaka::getPtrNative(modulesbuf.isInverted_buf);
      isLower = alpaka::getPtrNative(modulesbuf.isLower_buf);
      isAnchor = alpaka::getPtrNative(modulesbuf.isAnchor_buf);
      moduleType = alpaka::getPtrNative(modulesbuf.moduleType_buf);
      moduleLayerType = alpaka::getPtrNative(modulesbuf.moduleLayerType_buf);
      sdlLayers = alpaka::getPtrNative(modulesbuf.sdlLayers_buf);
      connectedPixels = alpaka::getPtrNative(modulesbuf.connectedPixels_buf);
    }
  };

  template <typename TDev>
  struct modulesBuffer : modules {
    Buf<TDev, unsigned int> detIds_buf;
    Buf<TDev, uint16_t> moduleMap_buf;
    Buf<TDev, unsigned int> mapdetId_buf;
    Buf<TDev, uint16_t> mapIdx_buf;
    Buf<TDev, uint16_t> nConnectedModules_buf;
    Buf<TDev, float> drdzs_buf;
    Buf<TDev, float> dxdys_buf;
    Buf<TDev, uint16_t> nModules_buf;
    Buf<TDev, uint16_t> nLowerModules_buf;
    Buf<TDev, uint16_t> partnerModuleIndices_buf;

    Buf<TDev, short> layers_buf;
    Buf<TDev, short> rings_buf;
    Buf<TDev, short> modules_buf;
    Buf<TDev, short> rods_buf;
    Buf<TDev, short> subdets_buf;
    Buf<TDev, short> sides_buf;
    Buf<TDev, float> eta_buf;
    Buf<TDev, float> r_buf;
    Buf<TDev, bool> isInverted_buf;
    Buf<TDev, bool> isLower_buf;
    Buf<TDev, bool> isAnchor_buf;
    Buf<TDev, ModuleType> moduleType_buf;
    Buf<TDev, ModuleLayerType> moduleLayerType_buf;
    Buf<TDev, int> sdlLayers_buf;
    Buf<TDev, unsigned int> connectedPixels_buf;

    modulesBuffer(TDev const& dev, unsigned int nMod, unsigned int nPixs)
        : detIds_buf(allocBufWrapper<unsigned int>(dev, nMod)),
          moduleMap_buf(allocBufWrapper<uint16_t>(dev, nMod * MAX_CONNECTED_MODULES)),
          mapdetId_buf(allocBufWrapper<unsigned int>(dev, nMod)),
          mapIdx_buf(allocBufWrapper<uint16_t>(dev, nMod)),
          nConnectedModules_buf(allocBufWrapper<uint16_t>(dev, nMod)),
          drdzs_buf(allocBufWrapper<float>(dev, nMod)),
          dxdys_buf(allocBufWrapper<float>(dev, nMod)),
          nModules_buf(allocBufWrapper<uint16_t>(dev, 1)),
          nLowerModules_buf(allocBufWrapper<uint16_t>(dev, 1)),
          partnerModuleIndices_buf(allocBufWrapper<uint16_t>(dev, nMod)),

          layers_buf(allocBufWrapper<short>(dev, nMod)),
          rings_buf(allocBufWrapper<short>(dev, nMod)),
          modules_buf(allocBufWrapper<short>(dev, nMod)),
          rods_buf(allocBufWrapper<short>(dev, nMod)),
          subdets_buf(allocBufWrapper<short>(dev, nMod)),
          sides_buf(allocBufWrapper<short>(dev, nMod)),
          eta_buf(allocBufWrapper<float>(dev, nMod)),
          r_buf(allocBufWrapper<float>(dev, nMod)),
          isInverted_buf(allocBufWrapper<bool>(dev, nMod)),
          isLower_buf(allocBufWrapper<bool>(dev, nMod)),
          isAnchor_buf(allocBufWrapper<bool>(dev, nMod)),
          moduleType_buf(allocBufWrapper<ModuleType>(dev, nMod)),
          moduleLayerType_buf(allocBufWrapper<ModuleLayerType>(dev, nMod)),
          sdlLayers_buf(allocBufWrapper<int>(dev, nMod)),
          connectedPixels_buf(allocBufWrapper<unsigned int>(dev, nPixs)) {
      setData(*this);
    }

    template <typename TQueue, typename TDevSrc>
    inline void copyFromSrc(TQueue queue, const modulesBuffer<TDevSrc>& src, bool isFull = true) {
      alpaka::memcpy(queue, detIds_buf, src.detIds_buf);
      if (isFull) {
        alpaka::memcpy(queue, moduleMap_buf, src.moduleMap_buf);
        alpaka::memcpy(queue, mapdetId_buf, src.mapdetId_buf);
        alpaka::memcpy(queue, mapIdx_buf, src.mapIdx_buf);
        alpaka::memcpy(queue, nConnectedModules_buf, src.nConnectedModules_buf);
        alpaka::memcpy(queue, drdzs_buf, src.drdzs_buf);
        alpaka::memcpy(queue, dxdys_buf, src.dxdys_buf);
      }
      alpaka::memcpy(queue, nModules_buf, src.nModules_buf);
      alpaka::memcpy(queue, nLowerModules_buf, src.nLowerModules_buf);
      if (isFull) {
        alpaka::memcpy(queue, partnerModuleIndices_buf, src.partnerModuleIndices_buf);
      }

      alpaka::memcpy(queue, layers_buf, src.layers_buf);
      alpaka::memcpy(queue, rings_buf, src.rings_buf);
      alpaka::memcpy(queue, modules_buf, src.modules_buf);
      alpaka::memcpy(queue, rods_buf, src.rods_buf);
      alpaka::memcpy(queue, subdets_buf, src.subdets_buf);
      alpaka::memcpy(queue, sides_buf, src.sides_buf);
      alpaka::memcpy(queue, eta_buf, src.eta_buf);
      alpaka::memcpy(queue, r_buf, src.r_buf);
      if (isFull) {
        alpaka::memcpy(queue, isInverted_buf, src.isInverted_buf);
      }
      alpaka::memcpy(queue, isLower_buf, src.isLower_buf);
      if (isFull) {
        alpaka::memcpy(queue, isAnchor_buf, src.isAnchor_buf);
      }
      alpaka::memcpy(queue, moduleType_buf, src.moduleType_buf);
      if (isFull) {
        alpaka::memcpy(queue, moduleLayerType_buf, src.moduleLayerType_buf);
        alpaka::memcpy(queue, sdlLayers_buf, src.sdlLayers_buf);
        alpaka::memcpy(queue, connectedPixels_buf, src.connectedPixels_buf);
      }
      alpaka::wait(queue);
    }

    template <typename TQueue>
    modulesBuffer(TQueue queue, const modulesBuffer<alpaka::DevCpu>& src, unsigned int nMod, unsigned int nPixs)
        : modulesBuffer(alpaka::getDev(queue), nMod, nPixs) {
      copyFromSrc(queue, src);
    }

    inline SDL::modules const* data() const { return this; }
  };

}  // namespace SDL
#endif
