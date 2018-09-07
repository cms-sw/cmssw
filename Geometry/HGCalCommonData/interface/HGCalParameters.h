#ifndef Geometry_HGCalCommonData_HGCalParameters_h
#define Geometry_HGCalCommonData_HGCalParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include <CLHEP/Geometry/Transform3D.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstdint>
#include <unordered_map>

class HGCalParameters {

public:

  typedef std::vector<std::unordered_map<int32_t,int32_t> > layer_map;
  typedef std::unordered_map<int32_t, int32_t>              wafer_map;

  static constexpr double   k_ScaleFromDDD = 0.1;
  static constexpr double   k_ScaleToDDD   = 10.0;
  static constexpr uint32_t k_CornerSize   = 6;

  struct hgtrap {
    int           lay;
    float         bl, tl, h, dz, alpha, cellSize;
  };

  struct hgtrform {
    int                zp, lay, sec, subsec;
    CLHEP::Hep3Vector  h3v;
    CLHEP::HepRotation hr;
  };
  
  HGCalParameters(const std::string& nam);
  ~HGCalParameters( void );
  void              fillModule(const hgtrap& mytr, bool reco);
  hgtrap            getModule(unsigned int k, bool reco) const;
  void              fillTrForm(const hgtrform& mytr);
  hgtrform          getTrForm(unsigned int k) const;
  void              addTrForm(const CLHEP::Hep3Vector& h3v);
  void              scaleTrForm(double);
  int               scintCells(const int layer) const
  { return nPhiBinBH_[scintType(layer)]; }
  double            scintCellSize(const int layer) const
  { return cellSize_[scintType(layer)]; }
  int               scintType(const int layer) const
  { return ((layer < layerFrontBH_[1]) ? 0 : 1); }
  std::array<int,4> getID(unsigned int k) const;

  std::string                     name_;
  int                             detectorType_;
  int                             nCells_;
  int                             nSectors_;
  std::vector<double>             cellSize_;
  std::vector<int>                moduleLayS_;
  std::vector<double>             moduleBlS_;
  std::vector<double>             moduleTlS_;
  std::vector<double>             moduleHS_;
  std::vector<double>             moduleDzS_;
  std::vector<double>             moduleAlphaS_;
  std::vector<double>             moduleCellS_;
  std::vector<int>                moduleLayR_;
  std::vector<double>             moduleBlR_;
  std::vector<double>             moduleTlR_;
  std::vector<double>             moduleHR_;
  std::vector<double>             moduleDzR_;
  std::vector<double>             moduleAlphaR_;
  std::vector<double>             moduleCellR_;
  std::vector<uint32_t>           trformIndex_;
  std::vector<double>             trformTranX_;
  std::vector<double>             trformTranY_;
  std::vector<double>             trformTranZ_;
  std::vector<double>             trformRotXX_;
  std::vector<double>             trformRotYX_;
  std::vector<double>             trformRotZX_;
  std::vector<double>             trformRotXY_;
  std::vector<double>             trformRotYY_;
  std::vector<double>             trformRotZY_;
  std::vector<double>             trformRotXZ_;
  std::vector<double>             trformRotYZ_;
  std::vector<double>             trformRotZZ_;
  std::vector<int>                layer_;
  std::vector<int>                layerIndex_;
  std::vector<int>                layerGroup_;
  std::vector<int>                cellFactor_; 
  std::vector<int>                depth_;
  std::vector<int>                depthIndex_;
  std::vector<int>                depthLayerF_;
  std::vector<double>             zLayerHex_;
  std::vector<double>             rMinLayHex_;
  std::vector<double>             rMaxLayHex_;
  std::vector<int>                waferCopy_;
  std::vector<int>                waferTypeL_;
  std::vector<int>                waferTypeT_;
  std::vector<double>             waferPosX_;
  std::vector<double>             waferPosY_;
  std::vector<double>             cellFineX_;
  std::vector<double>             cellFineY_;
  wafer_map                       cellFineIndex_;
  std::vector<bool>               cellFineHalf_;
  std::vector<double>             cellCoarseX_;
  std::vector<double>             cellCoarseY_;
  wafer_map                       cellCoarseIndex_;
  std::vector<bool>               cellCoarseHalf_;
  std::vector<int>                layerGroupM_;
  std::vector<int>                layerGroupO_;
  std::vector<double>             boundR_;
  std::vector<double>             rLimit_;
  std::vector<int>                cellFine_;
  std::vector<int>                cellCoarse_;
  double                          waferR_;
  std::vector<int>                levelT_;
  int                             levelZSide_;
  HGCalGeometryMode::GeometryMode mode_;
  std::vector<double>             slopeMin_;
  std::vector<double>             zFrontMin_;
  std::vector<double>             rMinFront_;
  layer_map                       copiesInLayers_;
  int                             nCellsFine_;
  int                             nCellsCoarse_;
  double                          waferSize_;
  double                          waferThick_;
  double                          sensorSeparation_;
  double                          mouseBite_;
  int                             waferUVMax_;
  std::vector<int>                waferUVMaxLayer_;
  bool                            defineFull_;
  std::vector<double>             cellThickness_;
  std::vector<double>             radius100to200_;
  std::vector<double>             radius200to300_;
  int                             choiceType_;
  int                             nCornerCut_;
  double                          fracAreaMin_;
  double                          zMinForRad_;
  std::vector<double>             radiusMixBoundary_;
  std::vector<int>                nPhiBinBH_;
  std::vector<int>                layerFrontBH_;
  std::vector<double>             rMinLayerBH_;
  std::vector<double>             radiusLayer_[2];
  std::vector<int>                iradMinBH_;
  std::vector<int>                iradMaxBH_;
  double                          minTileSize_;
  std::vector<int>                firstModule_;
  std::vector<int>                lastModule_;
  std::vector<double>             slopeTop_;
  std::vector<double>             zFrontTop_;
  std::vector<double>             rMaxFront_;
  std::vector<double>             zRanges_;
  int                             firstLayer_;
  int                             firstMixedLayer_;
  wafer_map                       wafersInLayers_;
  wafer_map                       typesInLayers_;
 
  COND_SERIALIZABLE;

private:

  const int kMaskZside   = 0x1;
  const int kMaskLayer   = 0x7F;
  const int kMaskSector  = 0x3FF;
  const int kMaskSubSec  = 0x1;
  const int kShiftZside  = 19;
  const int kShiftLayer  = 12;
  const int kShiftSector = 1;
  const int kShiftSubSec = 0;

};

#endif
