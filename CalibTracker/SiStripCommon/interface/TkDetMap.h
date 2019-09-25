#ifndef CalibTracker_SiStripCommon_TKHistoMap_h
#define CalibTracker_SiStripCommon_TKHistoMap_h

#include <vector>
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include <cstdint>

class TrackerTopology;

class TkLayerMap {
public:
  struct XYbin {
    XYbin(int16_t _ix = -999, int16_t _iy = -999, float _x = -999., float _y = -999.)
        : ix(_ix), iy(_iy), x(_x), y(_y) {}
    int16_t ix, iy;
    float x, y;
  };

  enum TkLayerEnum {
    INVALID = 0,
    TIB_L1,
    TIB_L2,
    TIB_L3,
    TIB_L4,  //1-4
    TIDM_D1,
    TIDM_D2,
    TIDM_D3,  //5-7
    TIDP_D1,
    TIDP_D2,
    TIDP_D3,  //8-10
    TOB_L1,
    TOB_L2,
    TOB_L3,
    TOB_L4,
    TOB_L5,
    TOB_L6,  //11-16
    TECM_W1,
    TECM_W2,
    TECM_W3,
    TECM_W4,
    TECM_W5,
    TECM_W6,
    TECM_W7,
    TECM_W8,
    TECM_W9,  //17-25
    TECP_W1,
    TECP_W2,
    TECP_W3,
    TECP_W4,
    TECP_W5,
    TECP_W6,
    TECP_W7,
    TECP_W8,
    TECP_W9,  //26-34
    NUMLAYERS
  };  //35

  TkLayerMap() {}

  TkLayerMap(int layer,
             std::size_t nchX,
             double lowX,
             double highX,
             std::size_t nchY,
             double lowY,
             double highY,
             const TrackerTopology* tTopo,
             const std::vector<uint32_t>& tkDetIdList,
             const std::vector<uint32_t>& singleExtString = {},
             const std::vector<uint32_t>& modulesInRingFront = {},
             const std::vector<uint32_t>& modulesInRingBack = {},
             const std::vector<uint32_t>& binForRing = {},
             uint32_t nstring_ext = 0,
             uint32_t nrod = 0)
      : layer_(layer),
        nchX_(nchX),
        lowX_(lowX),
        highX_(highX),
        nchY_(nchY),
        lowY_(lowY),
        highY_(highY),
        tTopo_(tTopo),
        singleExtStr_(singleExtString),
        modulesInRingFront_(modulesInRingFront),
        modulesInRingBack_(modulesInRingBack),
        binForRing_(binForRing),
        nStringExt_(nstring_ext),
        nRod_(nrod),
        binToDet_(std::vector<DetId>(std::size_t(nchX * nchY), 0)) {
    initMap(tkDetIdList);
  }

  TkLayerMap(int layer,
             std::size_t nchX,
             double lowX,
             double highX,
             std::size_t nchY,
             double lowY,
             double highY,
             const TrackerTopology* tTopo,
             const std::vector<uint32_t>& tkDetIdList,
             std::vector<uint32_t>&& singleExtString = {},
             std::vector<uint32_t>&& modulesInRingFront = {},
             std::vector<uint32_t>&& modulesInRingBack = {},
             std::vector<uint32_t>&& binForRing = {},
             uint32_t nstring_ext = 0,
             uint32_t nrod = 0)
      : layer_(layer),
        nchX_(nchX),
        lowX_(lowX),
        highX_(highX),
        nchY_(nchY),
        lowY_(lowY),
        highY_(highY),
        tTopo_(tTopo),
        singleExtStr_(singleExtString),
        modulesInRingFront_(modulesInRingFront),
        modulesInRingBack_(modulesInRingBack),
        binForRing_(binForRing),
        nStringExt_(nstring_ext),
        nRod_(nrod),
        binToDet_(std::vector<DetId>(std::size_t(nchX * nchY), 0)) {
    initMap(tkDetIdList);
  }

  void initMap(const std::vector<uint32_t>& tkDetIdList);

private:
  void initMap_TIB(const std::vector<uint32_t>& tkDetIdList);
  void initMap_TOB(const std::vector<uint32_t>& tkDetIdList);
  void initMap_TID(const std::vector<uint32_t>& tkDetIdList);
  void initMap_TEC(const std::vector<uint32_t>& tkDetIdList);

  std::size_t bin(std::size_t ix, std::size_t iy) const { return (ix - 1) + nchX_ * (iy - 1); }

public:
  static const int16_t layerSearch(DetId detid, const TrackerTopology* tTopo);

  std::size_t get_nchX() const { return nchX_; }
  std::size_t get_nchY() const { return nchY_; }
  double get_lowX() const { return lowX_; }
  double get_highX() const { return highX_; }
  double get_lowY() const { return lowY_; }
  double get_highY() const { return highY_; }
  const std::vector<DetId>& getBinToDet() const { return binToDet_; }

  DetId getDetFromBin(int ix, int iy) const {
    const auto idx = bin(ix, iy);
    return (idx < nchX_ * nchY_) ? binToDet_[idx] : DetId(0);
  }

  const XYbin getXY(DetId detid, int layerEnumNb = TkLayerMap::INVALID) const;

private:
  XYbin getXY_TIB(DetId detid) const;
  XYbin getXY_TOB(DetId detid) const;
  XYbin getXY_TID(DetId detid) const;
  XYbin getXY_TEC(DetId detid) const;

private:
  int layer_;  //In the enumerator sequence
  std::size_t nchX_;
  double lowX_, highX_;
  std::size_t nchY_;
  double lowY_, highY_;
  const TrackerTopology* tTopo_;

  std::vector<uint32_t> singleExtStr_;                                         // for TIB
  std::vector<uint32_t> modulesInRingFront_, modulesInRingBack_, binForRing_;  // for TEC
  uint32_t nStringExt_, nRod_;

  std::vector<DetId> binToDet_;
};

class TkDetMap {
public:
  TkDetMap(const TrackerTopology* tTopo) : tTopo_(tTopo) {
    TkMap.resize(TkLayerMap::NUMLAYERS);
  }  // maximal number of layers

  // modifiers
  void setLayerMap(int layer, const TkLayerMap& lyrMap) { TkMap[layer] = lyrMap; }
  void setLayerMap(int layer, TkLayerMap&& lyrMap) { TkMap[layer] = lyrMap; }

  // conversion
  static std::string getLayerName(int in);
  static int getLayerNum(const std::string& in);

  static void getSubDetLayerSide(int in, SiStripDetId::SubDetector&, uint32_t& layer, uint32_t& side);

  DetId getDetFromBin(int layer, int ix, int iy) const { return TkMap[layer].getDetFromBin(ix, iy); }
  DetId getDetFromBin(const std::string& layerName, int ix, int iy) const {
    return getDetFromBin(getLayerNum(layerName), ix, iy);
  }

  std::vector<DetId> getDetsForLayer(int layer) const {
    return TkMap[layer].getBinToDet();
  }  // const vector& -> vector conversion will copy

  // getXY and findLayer with caching, getComponents (for TkHistoMap)
  const TkLayerMap::XYbin& getXY(DetId detid,
                                 DetId& cached_detid,
                                 int16_t& cached_layer,
                                 TkLayerMap::XYbin& cached_XYbin) const;
  int16_t findLayer(DetId detid, DetId& cached_detid, int16_t& cached_layer, TkLayerMap::XYbin& cached_XYbin) const;
  void getComponents(int layer, int& nchX, double& lowX, double& highX, int& nchY, double& lowY, double& highY) const;

private:
  std::vector<TkLayerMap> TkMap;
  const TrackerTopology* tTopo_;
};

#endif
