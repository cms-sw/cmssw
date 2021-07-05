#ifndef HGCalCommonData_HGCalDDDConstants_h
#define HGCalCommonData_HGCalDDDConstants_h

/** \class HGCalDDDConstants
 *
 * this class reads the constant section of the numbering
 * xml-files of the  high granulairy calorimeter
 *
 *  $Date: 2014/03/20 00:06:50 $
 * \author Sunanda Banerjee, SINP <sunanda.banerjee@cern.ch>
 *
 */

#include <string>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

#include <unordered_map>

class HGCalDDDConstants {
public:
  HGCalDDDConstants(const HGCalParameters* hp, const std::string& name);
  ~HGCalDDDConstants();

  std::pair<int, int> assignCell(float x, float y, int lay, int subSec, bool reco) const;
  std::array<int, 5> assignCellHex(float x, float y, int lay, bool reco) const;
  std::array<int, 3> assignCellTrap(float x, float y, float z, int lay, bool reco) const;
  std::pair<double, double> cellEtaPhiTrap(int type, int irad) const;
  bool cellInLayer(int waferU, int waferV, int cellU, int cellV, int lay, bool reco) const;
  double cellSizeHex(int type) const;
  std::pair<double, double> cellSizeTrap(int type, int irad) const {
    return std::make_pair(hgpar_->radiusLayer_[type][irad - 1], hgpar_->radiusLayer_[type][irad]);
  }
  double cellThickness(int layer, int waferU, int waferV) const;
  HGCalTypes::CellType cellType(int type, int waferU, int waferV) const;
  double distFromEdgeHex(double x, double y, double z) const;
  double distFromEdgeTrap(double x, double y, double z) const;
  void etaPhiFromPosition(const double x,
                          const double y,
                          const double z,
                          const int layer,
                          int& ieta,
                          int& iphi,
                          int& type,
                          double& wt) const;
  int firstLayer() const { return hgpar_->firstLayer_; }
  HGCalGeometryMode::GeometryMode geomMode() const { return mode_; }
  int getLayer(double z, bool reco) const;
  int getLayerOffset() const { return hgpar_->layerOffset_; }
  HGCalParameters::hgtrap getModule(unsigned int k, bool hexType, bool reco) const;
  std::vector<HGCalParameters::hgtrap> getModules() const;
  const HGCalParameters* getParameter() const { return hgpar_; }
  int getPhiBins(int lay) const;
  std::pair<int, int> getREtaRange(int lay) const;
  const std::vector<double>& getRadiusLayer(int layer) const {
    int type = (tileTrapezoid() ? hgpar_->scintType(layer) : 0);
    return hgpar_->radiusLayer_[type];
  }
  HGCalParameters::hgtrform getTrForm(unsigned int k) const { return hgpar_->getTrForm(k); }
  unsigned int getTrFormN() const { return hgpar_->trformIndex_.size(); }
  std::vector<HGCalParameters::hgtrform> getTrForms() const;
  int getTypeTrap(int layer) const;
  int getTypeHex(int layer, int waferU, int waferV) const;
  std::pair<double, double> getXY(int layer, double x, double y, bool forwd) const;
  int getUVMax(int type) const { return ((type == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_); }
  bool isHalfCell(int waferType, int cell) const;
  bool isValidHex(int lay, int mod, int cell, bool reco) const;
  bool isValidHex8(int lay, int waferU, int waferV, bool fullAndPart = false) const;
  bool isValidHex8(int lay, int modU, int modV, int cellU, int cellV, bool fullAndPart = false) const;
  bool isValidTrap(int lay, int ieta, int iphi) const;
  int lastLayer(bool reco) const;
  int layerIndex(int lay, bool reco) const;
  unsigned int layers(bool reco) const;
  unsigned int layersInit(bool reco) const;
  std::pair<float, float> locateCell(int cell, int lay, int type, bool reco) const;
  std::pair<float, float> locateCell(
      int lay, int waferU, int waferV, int cellU, int cellV, bool reco, bool all, bool debug = false) const;
  std::pair<float, float> locateCell(const HGCSiliconDetId&, bool debug = false) const;
  std::pair<float, float> locateCell(const HGCScintillatorDetId&, bool debug = false) const;
  std::pair<float, float> locateCellHex(int cell, int wafer, bool reco) const;
  std::pair<float, float> locateCellTrap(int lay, int ieta, int iphi, bool reco) const;
  int levelTop(int ind = 0) const { return hgpar_->levelT_[ind]; }
  bool maskCell(const DetId& id, int corners) const;
  int maxCellUV() const { return (tileTrapezoid() ? hgpar_->nCellsFine_ : 2 * hgpar_->nCellsFine_); }
  int maxCells(bool reco) const;
  int maxCells(int lay, bool reco) const;
  int maxModules() const { return modHalf_; }
  int maxModulesPerLayer() const { return maxWafersPerLayer_; }
  int maxRows(int lay, bool reco) const;
  double minSlope() const { return hgpar_->slopeMin_[0]; }
  int modifyUV(int uv, int type1, int type2) const;
  int modules(int lay, bool reco) const;
  int modulesInit(int lay, bool reco) const;
  double mouseBite(bool reco) const;
  int numberCells(bool reco) const;
  std::vector<int> numberCells(int lay, bool reco) const;
  int numberCellsHexagon(int wafer) const;
  int numberCellsHexagon(int lay, int waferU, int waferV, bool flag) const;
  std::pair<double, double> rangeR(double z, bool reco) const;
  std::pair<double, double> rangeRLayer(int lay, bool reco) const;
  std::pair<double, double> rangeZ(bool reco) const;
  std::pair<int, int> rowColumnWafer(const int wafer) const;
  int sectors() const { return hgpar_->nSectors_; }
  std::pair<int, int> simToReco(int cell, int layer, int mod, bool half) const;
  int tileSiPM(int sipm) const { return ((sipm > 0) ? HGCalTypes::SiPMSmall : HGCalTypes::SiPMLarge); }
  bool tileTrapezoid() const {
    return ((mode_ == HGCalGeometryMode::Trapezoid) || (mode_ == HGCalGeometryMode::TrapezoidFile) ||
            (mode_ == HGCalGeometryMode::TrapezoidModule));
  }
  std::pair<int, int> tileType(int layer, int ring, int phi) const {
    int indx = HGCalTileIndex::tileIndex(layer, ring, phi);
    int type(-1), sipm(-1);
    auto itr = hgpar_->tileInfoMap_.find(indx);
    if (itr != hgpar_->tileInfoMap_.end()) {
      type = 1 + (itr->second).type;
      sipm = ((itr->second).sipm == HGCalTypes::SiPMLarge) ? 0 : 1;
    }
    return std::make_pair(type, sipm);
  }
  unsigned int volumes() const { return hgpar_->moduleLayR_.size(); }
  int waferFromCopy(int copy) const;
  void waferFromPosition(const double x, const double y, int& wafer, int& icell, int& celltyp) const;
  void waferFromPosition(const double x,
                         const double y,
                         const int layer,
                         int& waferU,
                         int& waferV,
                         int& cellU,
                         int& cellV,
                         int& celltype,
                         double& wt,
                         bool debug = false) const;
  bool waferHexagon6() const {
    return ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull));
  }
  bool waferHexagon8() const {
    return ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full) ||
            (mode_ == HGCalGeometryMode::Hexagon8File) || (mode_ == HGCalGeometryMode::Hexagon8Module));
  }
  bool waferInLayer(int wafer, int lay, bool reco) const;
  bool waferFullInLayer(int wafer, int lay, bool reco) const;
  int waferCount(const int type) const { return ((type == 0) ? waferMax_[2] : waferMax_[3]); }
  int waferMax() const { return waferMax_[1]; }
  int waferMin() const { return waferMax_[0]; }
  std::pair<double, double> waferParameters(bool reco) const;
  std::pair<double, double> waferPosition(int wafer, bool reco) const;
  std::pair<double, double> waferPosition(int lay, int waferU, int waferV, bool reco, bool debug = false) const;
  unsigned int waferFileSize() const { return hgpar_->waferInfoMap_.size(); }
  int waferFileIndex(unsigned int kk) const {
    if (kk < hgpar_->waferInfoMap_.size()) {
      auto itr = hgpar_->waferInfoMap_.begin();
      std::advance(itr, kk);
      return itr->first;
    } else
      return 0;
  }
  std::tuple<int, int, int> waferFileInfo(unsigned int kk) const {
    if (kk < hgpar_->waferInfoMap_.size()) {
      auto itr = hgpar_->waferInfoMap_.begin();
      std::advance(itr, kk);
      return std::make_tuple(itr->second.type, itr->second.part, itr->second.orient);
    } else
      return std::make_tuple(0, 0, 0);
  }
  std::tuple<int, int, int> waferFileInfoFromIndex(int kk) const {
    auto itr = hgpar_->waferInfoMap_.find(kk);
    if (itr != hgpar_->waferInfoMap_.end()) {
      return std::make_tuple(itr->second.type, itr->second.part, itr->second.orient);
    } else
      return std::make_tuple(0, 0, 0);
  }
  bool waferFileInfoExist(int kk) const { return (hgpar_->waferInfoMap_.find(kk) != hgpar_->waferInfoMap_.end()); }
  double waferSepar(bool reco) const {
    return (reco ? hgpar_->sensorSeparation_ : HGCalParameters::k_ScaleToDDD * hgpar_->sensorSeparation_);
  }
  double waferSize(bool reco) const {
    return (reco ? hgpar_->waferSize_ : HGCalParameters::k_ScaleToDDD * hgpar_->waferSize_);
  }
  int wafers() const;
  int wafers(int layer, int type) const;
  int waferToCopy(int wafer) const {
    return ((wafer >= 0) && (wafer < (int)(hgpar_->waferCopy_.size()))) ? hgpar_->waferCopy_[wafer]
                                                                        : (int)(hgpar_->waferCopy_.size());
  }
  // wafer transverse thickness classification (2 = coarse, 1 = fine)
  int waferTypeT(int wafer) const {
    return ((wafer >= 0) && (wafer < (int)(hgpar_->waferTypeT_.size()))) ? hgpar_->waferTypeT_[wafer] : 0;
  }
  // wafer longitudinal thickness classification (1 = 100um, 2 = 200um, 3=300um)
  int waferTypeL(int wafer) const {
    return ((wafer >= 0) && (wafer < (int)(hgpar_->waferTypeL_.size()))) ? hgpar_->waferTypeL_[wafer] : 0;
  }
  int waferType(DetId const& id, bool fromFile = false) const;
  int waferType(int layer, int waferU, int waferV, bool fromFile = false) const;
  std::tuple<int, int, int> waferType(HGCSiliconDetId const& id, bool fromFile = false) const;
  std::pair<int, int> waferTypeRotation(
      int layer, int waferU, int waferV, bool fromFile = false, bool debug = false) const;
  int waferUVMax() const { return hgpar_->waferUVMax_; }
  bool waferVirtual(int layer, int waferU, int waferV) const;
  double waferZ(int layer, bool reco) const;

private:
  int cellHex(double xx,
              double yy,
              const double& cellR,
              const std::vector<double>& posX,
              const std::vector<double>& posY) const;
  void cellHex(double xloc, double yloc, int cellType, int& cellU, int& cellV, bool debug = false) const;
  std::pair<int, float> getIndex(int lay, bool reco) const;
  int layerFromIndex(int index, bool reco) const;
  bool isValidCell(int layindex, int wafer, int cell) const;
  bool isValidCell8(int lay, int waferU, int waferV, int cellU, int cellV, int type) const;
  int32_t waferIndex(int wafer, int index) const;
  bool waferInLayerTest(int wafer, int lay, bool full) const;
  std::pair<double, double> waferPosition(int waferU, int waferV, bool reco) const;

  HGCalGeomTools geomTools_;
  const double k_horizontalShift = 1.0;
  const float dPhiMin = 0.02;
  typedef std::array<std::vector<int32_t>, 2> Simrecovecs;
  typedef std::array<int, 3> HGCWaferParam;
  const HGCalParameters* hgpar_;
  constexpr static double tan30deg_ = 0.5773502693;
  const double sqrt3_;
  double rmax_, hexside_;
  HGCalGeometryMode::GeometryMode mode_;
  bool fullAndPart_;
  int32_t tot_wafers_, modHalf_;
  std::array<uint32_t, 2> tot_layers_;
  Simrecovecs max_modules_layer_;
  int32_t maxWafersPerLayer_;
  std::map<int, HGCWaferParam> waferLayer_;
  std::array<int, 4> waferMax_;
  std::unordered_map<int32_t, bool> waferIn_;
};

#endif
