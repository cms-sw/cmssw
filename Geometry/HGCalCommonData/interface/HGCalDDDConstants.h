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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/HGCalCommonData/interface/HGCalCassette.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalCellUV.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include <CLHEP/Geometry/Point3D.h>

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

class HGCalDDDConstants {
public:
  HGCalDDDConstants(const HGCalParameters* hp, const std::string& name);
  ~HGCalDDDConstants() = default;

  std::pair<int, int> assignCell(float x, float y, int lay, int subSec, bool reco) const;
  std::array<int, 5> assignCellHex(float x, float y, int zside, int lay, bool reco, bool extend, bool debug) const;
  std::array<int, 3> assignCellTrap(float x, float y, float z, int lay, bool reco) const;
  bool cassetteMode() const {
    return ((mode_ == HGCalGeometryMode::Hexagon8Cassette) || (mode_ == HGCalGeometryMode::TrapezoidCassette) ||
            (mode_ == HGCalGeometryMode::Hexagon8CalibCell));
  }
  bool cassetteShiftScintillator(int zside, int layer, int iphi) const;
  bool cassetteShiftSilicon(int zside, int layer, int waferU, int waferV) const;
  int cassetteTile(int iphi) const {
    return (HGCalTileIndex::tileCassette(iphi, hgpar_->phiOffset_, hgpar_->nphiCassette_, hgpar_->cassettes_));
  }
  std::pair<double, double> cellEtaPhiTrap(int type, int irad) const;
  bool cellInLayer(int waferU, int waferV, int cellU, int cellV, int lay, int zside, bool reco) const;
  double cellSizeHex(int type) const;
  inline std::pair<double, double> cellSizeTrap(int type, int irad) const {
    return std::make_pair(hgpar_->radiusLayer_[type][irad - 1], hgpar_->radiusLayer_[type][irad]);
  }
  double cellThickness(int layer, int waferU, int waferV) const;
  int32_t cellType(int type, int waferU, int waferV, int iz, int fwdBack, int orient) const;
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
  inline int firstLayer() const { return hgpar_->firstLayer_; }
  inline HGCalGeometryMode::GeometryMode geomMode() const { return mode_; }
  int getLayer(double z, bool reco) const;
  int getLayerOffset() const { return hgpar_->layerOffset_; }
  HGCalParameters::hgtrap getModule(unsigned int k, bool hexType, bool reco) const;
  std::vector<HGCalParameters::hgtrap> getModules() const;
  inline const HGCalParameters* getParameter() const { return hgpar_; }
  int getPhiBins(int lay) const;
  std::pair<double, double> getRangeR(int, bool reco) const;
  std::pair<int, int> getREtaRange(int lay) const;
  inline const std::vector<double>& getRadiusLayer(int layer) const {
    return hgpar_->radiusLayer_[(tileTrapezoid() ? hgpar_->scintType(layer) : 0)];
  }
  inline HGCalParameters::hgtrform getTrForm(unsigned int k) const { return hgpar_->getTrForm(k); }
  inline unsigned int getTrFormN() const { return hgpar_->trformIndex_.size(); }
  std::vector<HGCalParameters::hgtrform> getTrForms() const;
  int getTypeTrap(int layer) const;
  int getTypeHex(int layer, int waferU, int waferV) const;
  std::pair<double, double> getXY(int layer, double x, double y, bool forwd) const;
  inline int getUVMax(int type) const { return ((type == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_); }
  bool isHalfCell(int waferType, int cell) const;
  bool isValidHex(int lay, int mod, int cell, bool reco) const;
  bool isValidHex8(int lay, int waferU, int waferV, bool fullAndPart) const;
  bool isValidHex8(int lay, int modU, int modV, int cellU, int cellV, bool fullAndPart) const;
  bool isValidTrap(int zside, int lay, int ieta, int iphi) const;
  int lastLayer(bool reco) const;
  int layerIndex(int lay, bool reco) const;
  unsigned int layers(bool reco) const;
  unsigned int layersInit(bool reco) const;
  inline int layerType(int lay) const {
    return ((hgpar_->layerType_.empty()) ? HGCalTypes::WaferCenter : hgpar_->layerType_[lay - hgpar_->firstLayer_]);
  }
  std::pair<float, float> localToGlobal8(
      int zside, int lay, int waferU, int waferV, double localX, double localY, bool reco, bool debug) const;
  std::pair<float, float> locateCell(int cell, int lay, int type, bool reco) const;
  std::pair<float, float> locateCell(
      int zside, int lay, int waferU, int waferV, int cellU, int cellV, bool reco, bool all, bool norot, bool debug)
      const;
  std::pair<float, float> locateCell(const HGCSiliconDetId&, bool debug) const;
  std::pair<float, float> locateCell(const HGCScintillatorDetId&, bool debug) const;
  std::pair<float, float> locateCellHex(int cell, int wafer, bool reco) const;
  std::pair<float, float> locateCellTrap(int zside, int lay, int ieta, int iphi, bool reco, bool debug) const;
  inline int levelTop(int ind = 0) const { return hgpar_->levelT_[ind]; }
  bool maskCell(const DetId& id, int corners) const;
  inline int maxCellUV() const { return (tileTrapezoid() ? hgpar_->nCellsFine_ : 2 * hgpar_->nCellsFine_); }
  int maxCells(bool reco) const;
  int maxCells(int lay, bool reco) const;
  inline int maxModules() const { return modHalf_; }
  inline int maxModulesPerLayer() const { return maxWafersPerLayer_; }
  int maxRows(int lay, bool reco) const;
  inline double minSlope() const { return hgpar_->slopeMin_[0]; }
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
  inline int sectors() const { return hgpar_->nSectors_; }
  std::pair<int, int> simToReco(int cell, int layer, int mod, bool half) const;
  int tileCount(int layer, int ring) const;
  bool tileExist(int zside, int layer, int ring, int phi) const;
  HGCalParameters::tileInfo tileInfo(int zside, int layer, int ring) const;
  bool tilePhiEdge(double phi, int layer, int iphi) const;
  bool tileRingEdge(double rho, int layer, int ring) const;
  std::pair<int, int> tileRings(int layer) const;
  inline int tileSiPM(int sipm) const { return ((sipm > 0) ? HGCalTypes::SiPMSmall : HGCalTypes::SiPMLarge); }
  bool tileTrapezoid() const {
    return ((mode_ == HGCalGeometryMode::Trapezoid) || (mode_ == HGCalGeometryMode::TrapezoidFile) ||
            (mode_ == HGCalGeometryMode::TrapezoidModule) || (mode_ == HGCalGeometryMode::TrapezoidCassette));
  }
  std::pair<int, int> tileType(int layer, int ring, int phi) const;
  inline bool trapezoidFile() const {
    return ((mode_ == HGCalGeometryMode::TrapezoidFile) || (mode_ == HGCalGeometryMode::TrapezoidModule) ||
            (mode_ == HGCalGeometryMode::TrapezoidCassette));
  }
  inline unsigned int volumes() const { return hgpar_->moduleLayR_.size(); }
  int waferFromCopy(int copy) const;
  void waferFromPosition(const double x, const double y, int& wafer, int& icell, int& celltyp) const;
  void waferFromPosition(const double x,
                         const double y,
                         const int zside,
                         const int layer,
                         int& waferU,
                         int& waferV,
                         int& cellU,
                         int& cellV,
                         int& celltype,
                         double& wt,
                         bool extend,
                         bool debug) const;
  inline bool waferHexagon6() const {
    return ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull));
  }
  inline bool waferHexagon8() const {
    return ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full) ||
            (mode_ == HGCalGeometryMode::Hexagon8File) || (mode_ == HGCalGeometryMode::Hexagon8Module) ||
            (mode_ == HGCalGeometryMode::Hexagon8Cassette) || (mode_ == HGCalGeometryMode::Hexagon8CalibCell));
  }
  inline bool waferHexagon8File() const {
    return ((mode_ == HGCalGeometryMode::Hexagon8File) || (mode_ == HGCalGeometryMode::Hexagon8Module) ||
            (mode_ == HGCalGeometryMode::Hexagon8Cassette) || (mode_ == HGCalGeometryMode::Hexagon8CalibCell));
  }
  inline bool waferHexagon8Module() const {
    return ((mode_ == HGCalGeometryMode::Hexagon8Module) || (mode_ == HGCalGeometryMode::Hexagon8Cassette) ||
            (mode_ == HGCalGeometryMode::Hexagon8CalibCell));
  }
  bool waferInLayer(int wafer, int lay, bool reco) const;
  bool waferFullInLayer(int wafer, int lay, bool reco) const;
  inline int waferCount(const int type) const { return ((type == 0) ? waferMax_[2] : waferMax_[3]); }
  HGCalParameters::waferInfo waferInfo(int lay, int waferU, int waferV) const;
  inline int waferMax() const { return waferMax_[1]; }
  inline int waferMin() const { return waferMax_[0]; }
  std::pair<double, double> waferParameters(bool reco) const;
  std::pair<double, double> waferPosition(int wafer, bool reco) const;
  std::pair<double, double> waferPosition(int lay, int waferU, int waferV, bool reco, bool debug) const;
  inline unsigned int waferFileSize() const { return hgpar_->waferInfoMap_.size(); }
  int waferFileIndex(unsigned int kk) const;
  std::tuple<int, int, int, int> waferFileInfo(unsigned int kk) const;
  std::tuple<int, int, int, int> waferFileInfoFromIndex(int kk) const;
  inline bool waferFileInfoExist(int kk) const {
    return (hgpar_->waferInfoMap_.find(kk) != hgpar_->waferInfoMap_.end());
  }
  GlobalPoint waferLocal2Global(
      HepGeom::Point3D<float>& loc, const DetId& id, bool useWafer, bool reco, bool debug) const;
  inline double waferSepar(bool reco) const {
    return (reco ? hgpar_->sensorSeparation_ : HGCalParameters::k_ScaleToDDD * hgpar_->sensorSeparation_);
  }
  inline double waferSize(bool reco) const {
    return (reco ? hgpar_->waferSize_ : HGCalParameters::k_ScaleToDDD * hgpar_->waferSize_);
  }
  int wafers() const;
  int wafers(int layer, int type) const;
  inline int waferToCopy(int wafer) const {
    return ((wafer >= 0) && (wafer < static_cast<int>(hgpar_->waferCopy_.size())))
               ? hgpar_->waferCopy_[wafer]
               : static_cast<int>(hgpar_->waferCopy_.size());
  }
  // wafer transverse thickness classification (2 = coarse, 1 = fine)
  inline int waferTypeT(int wafer) const {
    return ((wafer >= 0) && (wafer < static_cast<int>(hgpar_->waferTypeT_.size()))) ? hgpar_->waferTypeT_[wafer] : 0;
  }
  // wafer longitudinal thickness classification (1 = 100um, 2 = 200um, 3=300um)
  inline int waferTypeL(int wafer) const {
    return ((wafer >= 0) && (wafer < static_cast<int>(hgpar_->waferTypeL_.size()))) ? hgpar_->waferTypeL_[wafer] : 0;
  }
  int waferType(DetId const& id, bool fromFile) const;
  int waferType(int layer, int waferU, int waferV, bool fromFile) const;
  std::tuple<int, int, int> waferType(HGCSiliconDetId const& id, bool fromFile) const;
  std::pair<int, int> waferTypeRotation(int layer, int waferU, int waferV, bool fromFile, bool debug) const;
  inline int waferUVMax() const { return hgpar_->waferUVMax_; }
  bool waferVirtual(int layer, int waferU, int waferV) const;
  double waferZ(int layer, bool reco) const;

private:
  int cellHex(double xx,
              double yy,
              const double& cellR,
              const std::vector<double>& posX,
              const std::vector<double>& posY) const;
  void cellHex(
      double xloc, double yloc, int cellType, int place, int part, int& cellU, int& cellV, bool extend, bool debug)
      const;
  std::pair<int, float> getIndex(int lay, bool reco) const;
  int layerFromIndex(int index, bool reco) const;
  bool isValidCell(int layindex, int wafer, int cell) const;
  bool isValidCell8(int lay, int waferU, int waferV, int cellU, int cellV, int type) const;
  int32_t waferIndex(int wafer, int index) const;
  bool waferInLayerTest(int wafer, int lay, bool full) const;
  std::pair<double, double> waferPositionNoRot(int lay, int waferU, int waferV, bool reco, bool debug) const;
  std::pair<double, double> waferPosition(int waferU, int waferV, bool reco) const;

  HGCalCassette hgcassette_;
  std::unique_ptr<HGCalCell> hgcell_;
  std::unique_ptr<HGCalCellUV> hgcellUV_;
  HGCalGeomTools geomTools_;
  constexpr static double k_horizontalShift = 1.0;
  constexpr static float dPhiMin = 0.02;
  typedef std::array<std::vector<int32_t>, 2> Simrecovecs;
  typedef std::array<int, 3> HGCWaferParam;
  const HGCalParameters* hgpar_;
  constexpr static double tan30deg_ = 0.5773502693;
  constexpr static double tol_ = 0.001;
  const double sqrt3_;
  const HGCalGeometryMode::GeometryMode mode_;
  const bool fullAndPart_;
  double rmax_, hexside_;
  double rmaxT_, hexsideT_;
  int32_t tot_wafers_, modHalf_;
  std::array<uint32_t, 2> tot_layers_;
  Simrecovecs max_modules_layer_;
  int32_t maxWafersPerLayer_;
  std::map<int, HGCWaferParam> waferLayer_;
  std::array<int, 4> waferMax_;
  std::unordered_map<int32_t, bool> waferIn_;
};

#endif
