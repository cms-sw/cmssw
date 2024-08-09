#ifndef RecoTracker_LSTCore_interface_Module_h
#define RecoTracker_LSTCore_interface_Module_h

#include "RecoTracker/LSTCore/interface/Constants.h"

namespace lst {
  enum SubDet { InnerPixel = 0, Barrel = 5, Endcap = 4 };

  enum Side { NegZ = 1, PosZ = 2, Center = 3 };

  enum ModuleType { PS, TwoS, PixelModule };

  enum ModuleLayerType { Pixel, Strip, InnerPixelLayer };

  struct Modules {
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
    const int* lstLayers;
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
    void setData(TBuff const& buf) {
      detIds = alpaka::getPtrNative(buf.detIds_buf);
      moduleMap = alpaka::getPtrNative(buf.moduleMap_buf);
      mapdetId = alpaka::getPtrNative(buf.mapdetId_buf);
      mapIdx = alpaka::getPtrNative(buf.mapIdx_buf);
      nConnectedModules = alpaka::getPtrNative(buf.nConnectedModules_buf);
      drdzs = alpaka::getPtrNative(buf.drdzs_buf);
      dxdys = alpaka::getPtrNative(buf.dxdys_buf);
      nModules = alpaka::getPtrNative(buf.nModules_buf);
      nLowerModules = alpaka::getPtrNative(buf.nLowerModules_buf);
      partnerModuleIndices = alpaka::getPtrNative(buf.partnerModuleIndices_buf);

      layers = alpaka::getPtrNative(buf.layers_buf);
      rings = alpaka::getPtrNative(buf.rings_buf);
      modules = alpaka::getPtrNative(buf.modules_buf);
      rods = alpaka::getPtrNative(buf.rods_buf);
      subdets = alpaka::getPtrNative(buf.subdets_buf);
      sides = alpaka::getPtrNative(buf.sides_buf);
      eta = alpaka::getPtrNative(buf.eta_buf);
      r = alpaka::getPtrNative(buf.r_buf);
      isInverted = alpaka::getPtrNative(buf.isInverted_buf);
      isLower = alpaka::getPtrNative(buf.isLower_buf);
      isAnchor = alpaka::getPtrNative(buf.isAnchor_buf);
      moduleType = alpaka::getPtrNative(buf.moduleType_buf);
      moduleLayerType = alpaka::getPtrNative(buf.moduleLayerType_buf);
      lstLayers = alpaka::getPtrNative(buf.lstLayers_buf);
      connectedPixels = alpaka::getPtrNative(buf.connectedPixels_buf);
    }
  };

  template <typename TDev>
  struct ModulesBuffer {
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
    Buf<TDev, int> lstLayers_buf;
    Buf<TDev, unsigned int> connectedPixels_buf;

    Modules data_;

    ModulesBuffer(TDev const& dev, unsigned int nMod, unsigned int nPixs)
        : detIds_buf(allocBufWrapper<unsigned int>(dev, nMod)),
          moduleMap_buf(allocBufWrapper<uint16_t>(dev, nMod * max_connected_modules)),
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
          lstLayers_buf(allocBufWrapper<int>(dev, nMod)),
          connectedPixels_buf(allocBufWrapper<unsigned int>(dev, nPixs)) {
      data_.setData(*this);
    }

    template <typename TQueue, typename TDevSrc>
    inline void copyFromSrc(TQueue queue, ModulesBuffer<TDevSrc> const& src, bool isFull = true) {
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
        alpaka::memcpy(queue, lstLayers_buf, src.lstLayers_buf);
        alpaka::memcpy(queue, connectedPixels_buf, src.connectedPixels_buf);
      }
      alpaka::wait(queue);
    }

    template <typename TQueue, typename TDevSrc>
    ModulesBuffer(TQueue queue, ModulesBuffer<TDevSrc> const& src, unsigned int nMod, unsigned int nPixs)
        : ModulesBuffer(alpaka::getDev(queue), nMod, nPixs) {
      copyFromSrc(queue, src);
    }

    inline Modules const* data() const { return &data_; }
  };

}  // namespace lst
#endif
