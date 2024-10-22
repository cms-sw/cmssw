#ifndef HGCalCommonData_HGCalTBDDDConstants_h
#define HGCalCommonData_HGCalTBDDDConstants_h

/** \class HGCalTBDDDConstants
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
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBParameters.h"
#include <CLHEP/Geometry/Point3D.h>

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

class HGCalTBDDDConstants {
public:
  HGCalTBDDDConstants(const HGCalTBParameters* hp, const std::string& name);
  ~HGCalTBDDDConstants() = default;

  std::pair<int, int> assignCell(float x, float y, int lay, int subSec, bool reco) const;
  double cellSizeHex(int type) const;
  double cellThickness(int layer, int wafer) const;
  double distFromEdgeHex(double x, double y, double z) const;
  inline int firstLayer() const { return hgpar_->firstLayer_; }
  inline HGCalGeometryMode::GeometryMode geomMode() const { return mode_; }
  int getLayer(double z, bool reco) const;
  HGCalTBParameters::hgtrap getModule(unsigned int k, bool hexType, bool reco) const;
  std::vector<HGCalTBParameters::hgtrap> getModules() const;
  inline const HGCalTBParameters* getParameter() const { return hgpar_; }
  inline HGCalTBParameters::hgtrform getTrForm(unsigned int k) const { return hgpar_->getTrForm(k); }
  inline unsigned int getTrFormN() const { return hgpar_->trformIndex_.size(); }
  std::vector<HGCalTBParameters::hgtrform> getTrForms() const;
  int getTypeHex(int layer, int wafer) const { return -1; }
  bool isHalfCell(int waferType, int cell) const;
  bool isValidHex(int lay, int mod, int cell, bool reco) const;
  int lastLayer(bool reco) const;
  int layerIndex(int lay, bool reco) const;
  unsigned int layers(bool reco) const;
  unsigned int layersInit(bool reco) const;
  inline int layerType(int lay) const { return HGCalTypes::WaferCenter; }
  std::pair<float, float> locateCell(int cell, int lay, int type, bool reco) const;
  std::pair<float, float> locateCellHex(int cell, int wafer, bool reco) const;
  inline int levelTop(int ind = 0) const { return hgpar_->levelT_[ind]; }
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
  std::pair<double, double> rangeR(double z, bool reco) const;
  std::pair<double, double> rangeRLayer(int lay, bool reco) const;
  std::pair<double, double> rangeZ(bool reco) const;
  std::pair<int, int> rowColumnWafer(const int wafer) const;
  inline int sectors() const { return hgpar_->nSectors_; }
  std::pair<int, int> simToReco(int cell, int layer, int mod, bool half) const;
  inline unsigned int volumes() const { return hgpar_->moduleLayR_.size(); }
  int waferFromCopy(int copy) const;
  void waferFromPosition(const double x, const double y, int& wafer, int& icell, int& celltyp) const;
  inline bool waferHexagon6() const {
    return ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull));
  }
  bool waferInLayer(int wafer, int lay, bool reco) const;
  bool waferFullInLayer(int wafer, int lay, bool reco) const;
  inline int waferCount(const int type) const { return ((type == 0) ? waferMax_[2] : waferMax_[3]); }
  inline int waferMax() const { return waferMax_[1]; }
  inline int waferMin() const { return waferMax_[0]; }
  std::pair<double, double> waferParameters(bool reco) const;
  std::pair<double, double> waferPosition(int wafer, bool reco) const;
  GlobalPoint waferLocal2Global(
      HepGeom::Point3D<float>& loc, const DetId& id, bool useWafer, bool reco, bool debug) const;
  inline double waferSepar(bool reco) const {
    return (reco ? hgpar_->sensorSeparation_ : HGCalTBParameters::k_ScaleToDDD * hgpar_->sensorSeparation_);
  }
  inline double waferSize(bool reco) const {
    return (reco ? hgpar_->waferSize_ : HGCalTBParameters::k_ScaleToDDD * hgpar_->waferSize_);
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
  int waferType(DetId const& id) const;
  int waferType(int layer, int wafer) const;
  std::tuple<int, int, int> waferType(HGCSiliconDetId const& id, bool fromFile = false) const;
  inline int waferUVMax() const { return hgpar_->waferUVMax_; }
  bool waferVirtual(int layer, int wafer) const;
  double waferZ(int layer, bool reco) const;

private:
  int cellHex(double xx,
              double yy,
              const double& cellR,
              const std::vector<double>& posX,
              const std::vector<double>& posY) const;
  std::pair<int, float> getIndex(int lay, bool reco) const;
  int layerFromIndex(int index, bool reco) const;
  bool isValidCell(int layindex, int wafer, int cell) const;
  int32_t waferIndex(int wafer, int index) const;
  bool waferInLayerTest(int wafer, int lay) const { return waferHexagon6(); }

  HGCalGeomTools geomTools_;
  const double k_horizontalShift = 1.0;
  const float dPhiMin = 0.02;
  typedef std::array<std::vector<int32_t>, 2> Simrecovecs;
  typedef std::array<int, 3> HGCWaferParam;
  const HGCalTBParameters* hgpar_;
  constexpr static double tan30deg_ = 0.5773502693;
  const double sqrt3_;
  const HGCalGeometryMode::GeometryMode mode_;
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
