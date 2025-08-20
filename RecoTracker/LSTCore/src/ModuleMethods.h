#ifndef RecoTracker_LSTCore_src_ModuleMethods_h
#define RecoTracker_LSTCore_src_ModuleMethods_h

#include <map>
#include <iostream>

#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/ModulesSoA.h"
#include "RecoTracker/LSTCore/interface/ModulesHostCollection.h"
#include "RecoTracker/LSTCore/interface/TiltedGeometry.h"
#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"
#include "RecoTracker/LSTCore/interface/ModuleConnectionMap.h"
#include "RecoTracker/LSTCore/interface/PixelMap.h"

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace lst {
  struct ModuleMetaData {
    std::map<unsigned int, uint16_t> detIdToIndex;
    std::map<unsigned int, float> module_x;
    std::map<unsigned int, float> module_y;
    std::map<unsigned int, float> module_z;
    std::map<unsigned int, unsigned int> module_type;  // 23 : Ph2PSP, 24 : Ph2PSS, 25 : Ph2SS
    // https://github.com/cms-sw/cmssw/blob/5e809e8e0a625578aa265dc4b128a93830cb5429/Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h#L29
  };

  bool parseIsLower(bool isInvertedx, unsigned int detId) { return (isInvertedx) ? !(detId & 1) : (detId & 1); }

  unsigned int parsePartnerModuleId(unsigned int detId, bool isLowerx, bool isInvertedx) {
    return isLowerx ? (isInvertedx ? detId - 1 : detId + 1) : (isInvertedx ? detId + 1 : detId - 1);
  }

  bool parseIsInverted(short subdet, short side, short module, short layer) {
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
  }

  inline std::tuple<unsigned int,
                    std::vector<unsigned int>,
                    unsigned int,
                    std::vector<unsigned int>,
                    unsigned int,
                    std::vector<unsigned int>>
  getConnectedPixels(uint16_t nModules, unsigned int& nPixels, PixelMap& pixelMapping, MapPLStoLayer const& pLStoLayer) {
    std::vector<unsigned int> connectedModuleDetIds;
    std::vector<unsigned int> connectedModuleDetIds_pos;
    std::vector<unsigned int> connectedModuleDetIds_neg;

    unsigned int totalSizes = 0;
    unsigned int totalSizes_pos = 0;
    unsigned int totalSizes_neg = 0;
    for (unsigned int isuperbin = 0; isuperbin < size_superbins; isuperbin++) {
      int sizes = 0;
      for (auto const& mCM_pLS : pLStoLayer[0]) {
        std::vector<unsigned int> connectedModuleDetIds_pLS =
            mCM_pLS.getConnectedModuleDetIds(isuperbin + size_superbins);
        connectedModuleDetIds.insert(
            connectedModuleDetIds.end(), connectedModuleDetIds_pLS.begin(), connectedModuleDetIds_pLS.end());
        sizes += connectedModuleDetIds_pLS.size();
      }
      pixelMapping.connectedPixelsIndex[isuperbin] = totalSizes;
      pixelMapping.connectedPixelsSizes[isuperbin] = sizes;
      totalSizes += sizes;

      int sizes_pos = 0;
      for (auto const& mCM_pLS : pLStoLayer[1]) {
        std::vector<unsigned int> connectedModuleDetIds_pLS_pos = mCM_pLS.getConnectedModuleDetIds(isuperbin);
        connectedModuleDetIds_pos.insert(connectedModuleDetIds_pos.end(),
                                         connectedModuleDetIds_pLS_pos.begin(),
                                         connectedModuleDetIds_pLS_pos.end());
        sizes_pos += connectedModuleDetIds_pLS_pos.size();
      }
      pixelMapping.connectedPixelsIndexPos[isuperbin] = totalSizes_pos;
      pixelMapping.connectedPixelsSizesPos[isuperbin] = sizes_pos;
      totalSizes_pos += sizes_pos;

      int sizes_neg = 0;
      for (auto const& mCM_pLS : pLStoLayer[2]) {
        std::vector<unsigned int> connectedModuleDetIds_pLS_neg = mCM_pLS.getConnectedModuleDetIds(isuperbin);
        connectedModuleDetIds_neg.insert(connectedModuleDetIds_neg.end(),
                                         connectedModuleDetIds_pLS_neg.begin(),
                                         connectedModuleDetIds_pLS_neg.end());
        sizes_neg += connectedModuleDetIds_pLS_neg.size();
      }
      pixelMapping.connectedPixelsIndexNeg[isuperbin] = totalSizes_neg;
      pixelMapping.connectedPixelsSizesNeg[isuperbin] = sizes_neg;
      totalSizes_neg += sizes_neg;
    }

    nPixels = totalSizes + totalSizes_pos + totalSizes_neg;

    return {totalSizes,
            connectedModuleDetIds,
            totalSizes_pos,
            connectedModuleDetIds_pos,
            totalSizes_neg,
            connectedModuleDetIds_neg};
  }

  inline void fillConnectedModuleArrayExplicit(Modules modules,
                                               ModuleMetaData const& mmd,
                                               ModuleConnectionMap const& moduleConnectionMap) {
    Params_Modules::ArrayU16xMaxConnected* moduleMap = modules.moduleMap();
    uint16_t* nConnectedModules = modules.nConnectedModules();

    for (auto it = mmd.detIdToIndex.begin(); it != mmd.detIdToIndex.end(); ++it) {
      unsigned int detId = it->first;
      uint16_t index = it->second;
      auto& connectedModules = moduleConnectionMap.getConnectedModuleDetIds(detId);
      nConnectedModules[index] = connectedModules.size();
      for (uint16_t i = 0; i < nConnectedModules[index]; i++) {
        moduleMap[index][i] = mmd.detIdToIndex.at(connectedModules[i]);
      }
    }
  }

  inline void fillMapArraysExplicit(Modules modules, ModuleMetaData const& mmd) {
    uint16_t* mapIdx = modules.mapIdx();
    unsigned int* mapdetId = modules.mapdetId();

    unsigned int counter = 0;
    for (auto it = mmd.detIdToIndex.begin(); it != mmd.detIdToIndex.end(); ++it) {
      unsigned int detId = it->first;
      unsigned int index = it->second;
      mapIdx[counter] = index;
      mapdetId[counter] = detId;
      counter++;
    }
  }

  inline void setDerivedQuantities(unsigned int detId,
                                   unsigned short& layer,
                                   unsigned short& ring,
                                   unsigned short& rod,
                                   unsigned short& module,
                                   unsigned short& subdet,
                                   unsigned short& side,
                                   float m_x,
                                   float m_y,
                                   float m_z,
                                   float& eta,
                                   float& r) {
    subdet = (detId & (7 << 25)) >> 25;
    side = (subdet == Endcap) ? (detId & (3 << 23)) >> 23 : (detId & (3 << 18)) >> 18;
    layer = (subdet == Endcap) ? (detId & (7 << 18)) >> 18 : (detId & (7 << 20)) >> 20;
    ring = (subdet == Endcap) ? (detId & (15 << 12)) >> 12 : 0;
    module = (detId & (127 << 2)) >> 2;
    rod = (subdet == Endcap) ? 0 : (detId & (127 << 10)) >> 10;

    r = std::sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
    eta = ((m_z > 0) - (m_z < 0)) * std::acosh(r / std::sqrt(m_x * m_x + m_y * m_y));
  }

  inline void loadCentroidsFromFile(const char* filePath, ModuleMetaData& mmd, uint16_t& nModules) {
    std::ifstream ifile(filePath, std::ios::binary);
    if (!ifile.is_open()) {
      throw std::runtime_error("Unable to open file: " + std::string(filePath));
    }

    uint16_t counter = 0;
    while (!ifile.eof()) {
      unsigned int temp_detId;
      float module_x, module_y, module_z;
      int module_type;

      ifile.read(reinterpret_cast<char*>(&temp_detId), sizeof(temp_detId));
      ifile.read(reinterpret_cast<char*>(&module_x), sizeof(module_x));
      ifile.read(reinterpret_cast<char*>(&module_y), sizeof(module_y));
      ifile.read(reinterpret_cast<char*>(&module_z), sizeof(module_z));
      ifile.read(reinterpret_cast<char*>(&module_type), sizeof(module_type));

      if (ifile) {
        mmd.detIdToIndex[temp_detId] = counter;
        mmd.module_x[temp_detId] = module_x;
        mmd.module_y[temp_detId] = module_y;
        mmd.module_z[temp_detId] = module_z;
        mmd.module_type[temp_detId] = module_type;
        counter++;
      } else {
        if (!ifile.eof()) {
          throw std::runtime_error("Failed to read data for detId: " + std::to_string(temp_detId));
        }
      }
    }

    mmd.detIdToIndex[1] = counter;  //pixel module is the last module in the module list
    counter++;
    nModules = counter;
  }

  inline std::shared_ptr<ModulesHostCollection> loadModulesFromFile(MapPLStoLayer const& pLStoLayer,
                                                                    const char* moduleMetaDataFilePath,
                                                                    uint16_t& nModules,
                                                                    uint16_t& nLowerModules,
                                                                    unsigned int& nPixels,
                                                                    PixelMap& pixelMapping,
                                                                    const EndcapGeometry& endcapGeometry,
                                                                    const TiltedGeometry& tiltedGeometry,
                                                                    const ModuleConnectionMap& moduleConnectionMap) {
    ModuleMetaData mmd;

    loadCentroidsFromFile(moduleMetaDataFilePath, mmd, nModules);

    // TODO: this whole section could use some refactoring
    auto [totalSizes,
          connectedModuleDetIds,
          totalSizes_pos,
          connectedModuleDetIds_pos,
          totalSizes_neg,
          connectedModuleDetIds_neg] = getConnectedPixels(nModules, nPixels, pixelMapping, pLStoLayer);

    std::array<int, 2> const modules_sizes{{static_cast<int>(nModules), static_cast<int>(nPixels)}};

    auto modulesHC = std::make_shared<ModulesHostCollection>(modules_sizes, cms::alpakatools::host());

    auto modules_view = modulesHC->view<ModulesSoA>();

    // Getting the underlying data pointers
    unsigned int* host_detIds = modules_view.detIds();
    short* host_layers = modules_view.layers();
    short* host_rings = modules_view.rings();
    short* host_rods = modules_view.rods();
    short* host_modules = modules_view.modules();
    short* host_subdets = modules_view.subdets();
    short* host_sides = modules_view.sides();
    float* host_eta = modules_view.eta();
    float* host_r = modules_view.r();
    bool* host_isInverted = modules_view.isInverted();
    bool* host_isLower = modules_view.isLower();
    bool* host_isAnchor = modules_view.isAnchor();
    ModuleType* host_moduleType = modules_view.moduleType();
    ModuleLayerType* host_moduleLayerType = modules_view.moduleLayerType();
    float* host_dxdys = modules_view.dxdys();
    float* host_drdzs = modules_view.drdzs();
    uint16_t* host_nModules = &modules_view.nModules();
    uint16_t* host_nLowerModules = &modules_view.nLowerModules();
    uint16_t* host_partnerModuleIndices = modules_view.partnerModuleIndices();
    int* host_lstLayers = modules_view.lstLayers();

    //reassign detIdToIndex indices here
    nLowerModules = (nModules - 1) / 2;
    uint16_t lowerModuleCounter = 0;
    uint16_t upperModuleCounter = nLowerModules + 1;
    //0 to nLowerModules - 1 => only lower modules, nLowerModules - pixel module, nLowerModules + 1 to nModules => upper modules
    for (auto it = mmd.detIdToIndex.begin(); it != mmd.detIdToIndex.end(); it++) {
      unsigned int detId = it->first;
      float m_x = mmd.module_x[detId];
      float m_y = mmd.module_y[detId];
      float m_z = mmd.module_z[detId];
      unsigned int m_t = mmd.module_type[detId];

      float eta, r;

      uint16_t index;
      unsigned short layer, ring, rod, module, subdet, side;
      bool isInverted, isLower;
      if (detId == 1) {
        layer = 0;
        ring = 0;
        rod = 0;
        module = 0;
        subdet = 0;
        side = 0;
        isInverted = false;
        isLower = false;
        eta = 0;
        r = 0;
      } else {
        setDerivedQuantities(detId, layer, ring, rod, module, subdet, side, m_x, m_y, m_z, eta, r);
        isInverted = parseIsInverted(subdet, side, module, layer);
        isLower = parseIsLower(isInverted, detId);
      }
      if (isLower) {
        index = lowerModuleCounter;
        lowerModuleCounter++;
      } else if (detId != 1) {
        index = upperModuleCounter;
        upperModuleCounter++;
      } else {
        index = nLowerModules;  //pixel
      }
      //reassigning indices!
      mmd.detIdToIndex[detId] = index;
      host_detIds[index] = detId;
      host_layers[index] = layer;
      host_rings[index] = ring;
      host_rods[index] = rod;
      host_modules[index] = module;
      host_subdets[index] = subdet;
      host_sides[index] = side;
      host_eta[index] = eta;
      host_r[index] = r;
      host_isInverted[index] = isInverted;
      host_isLower[index] = isLower;

      //assigning other variables!
      if (detId == 1) {
        host_moduleType[index] = PixelModule;
        host_moduleLayerType[index] = lst::InnerPixelLayer;
        host_dxdys[index] = 0;
        host_drdzs[index] = 0;
        host_isAnchor[index] = false;
      } else {
        host_moduleType[index] = (m_t == 25 ? lst::TwoS : lst::PS);
        host_moduleLayerType[index] = (m_t == 23 ? lst::Pixel : lst::Strip);

        if ((host_moduleType[index] == lst::PS and host_moduleLayerType[index] == lst::Pixel) ||
            (host_moduleType[index] == lst::TwoS and host_isLower[index])) {
          host_isAnchor[index] = true;
        } else {
          host_isAnchor[index] = false;
        }

        host_dxdys[index] = (subdet == Endcap) ? endcapGeometry.getdxdy_slope(detId) : tiltedGeometry.getDxDy(detId);
        host_drdzs[index] = (subdet == Barrel) ? tiltedGeometry.getDrDz(detId) : 0;
      }

      host_lstLayers[index] =
          layer + 6 * (subdet == lst::Endcap) + 5 * (subdet == lst::Endcap and host_moduleType[index] == lst::TwoS);
    }

    //partner module stuff, and slopes and drdz move around
    for (auto it = mmd.detIdToIndex.begin(); it != mmd.detIdToIndex.end(); it++) {
      auto& detId = it->first;
      auto& index = it->second;
      if (detId != 1) {
        host_partnerModuleIndices[index] =
            mmd.detIdToIndex[parsePartnerModuleId(detId, host_isLower[index], host_isInverted[index])];
        //add drdz and slope importing stuff here!
        if (host_drdzs[index] == 0) {
          host_drdzs[index] = host_drdzs[host_partnerModuleIndices[index]];
        }
        if (host_dxdys[index] == 0) {
          host_dxdys[index] = host_dxdys[host_partnerModuleIndices[index]];
        }
      }
    }

    *host_nModules = nModules;
    *host_nLowerModules = nLowerModules;

    // Fill pixel part
    pixelMapping.pixelModuleIndex = mmd.detIdToIndex.at(1);

    auto modulesPixel_view = modulesHC->view<ModulesPixelSoA>();
    auto connectedPixels =
        cms::alpakatools::make_host_view(modulesPixel_view.connectedPixels(), modulesPixel_view.metadata().size());
    for (unsigned int icondet = 0; icondet < totalSizes; icondet++) {
      connectedPixels[icondet] = mmd.detIdToIndex.at(connectedModuleDetIds[icondet]);
    }
    for (unsigned int icondet = 0; icondet < totalSizes_pos; icondet++) {
      connectedPixels[icondet + totalSizes] = mmd.detIdToIndex.at(connectedModuleDetIds_pos[icondet]);
    }
    for (unsigned int icondet = 0; icondet < totalSizes_neg; icondet++) {
      connectedPixels[icondet + totalSizes + totalSizes_pos] = mmd.detIdToIndex.at(connectedModuleDetIds_neg[icondet]);
    }

    fillConnectedModuleArrayExplicit(modules_view, mmd, moduleConnectionMap);
    fillMapArraysExplicit(modules_view, mmd);

    return modulesHC;
  }
}  // namespace lst
#endif
