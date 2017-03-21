#ifndef Geometry_HGCalCommonData_HGCalParameters_h
#define Geometry_HGCalCommonData_HGCalParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <CLHEP/Geometry/Transform3D.h>
#include <string>
#include <vector>
#include <iostream>
#include <stdint.h>

#include<vector>
#include<unordered_map>

class HGCalParameters {

public:

  typedef std::vector<std::unordered_map<int32_t,int32_t> > layer_map;

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
  void     fillModule(const hgtrap& mytr, bool reco);
  hgtrap   getModule(unsigned int k, bool reco) const;
  void     fillTrForm(const hgtrform& mytr);
  hgtrform getTrForm(unsigned int k) const;
  void     addTrForm(const CLHEP::Hep3Vector& h3v);
  void     scaleTrForm(double);

  static const int kMaskZside   = 0x1;
  static const int kMaskLayer   = 0x7F;
  static const int kMaskSector  = 0x3FF;
  static const int kMaskSubSec  = 0x1;
  static const int kShiftZside  = 19;
  static const int kShiftLayer  = 12;
  static const int kShiftSector = 1;
  static const int kShiftSubSec = 0;

  std::string              name_;
  int                      nCells_;
  int                      nSectors_;
  std::vector<double>      cellSize_;
  std::vector<int>         moduleLayS_;
  std::vector<double>      moduleBlS_;
  std::vector<double>      moduleTlS_;
  std::vector<double>      moduleHS_;
  std::vector<double>      moduleDzS_;
  std::vector<double>      moduleAlphaS_;
  std::vector<double>      moduleCellS_;
  std::vector<int>         moduleLayR_;
  std::vector<double>      moduleBlR_;
  std::vector<double>      moduleTlR_;
  std::vector<double>      moduleHR_;
  std::vector<double>      moduleDzR_;
  std::vector<double>      moduleAlphaR_;
  std::vector<double>      moduleCellR_;
  std::vector<uint32_t>    trformIndex_;
  std::vector<double>      trformTranX_;
  std::vector<double>      trformTranY_;
  std::vector<double>      trformTranZ_;
  std::vector<double>      trformRotXX_;
  std::vector<double>      trformRotYX_;
  std::vector<double>      trformRotZX_;
  std::vector<double>      trformRotXY_;
  std::vector<double>      trformRotYY_;
  std::vector<double>      trformRotZY_;
  std::vector<double>      trformRotXZ_;
  std::vector<double>      trformRotYZ_;
  std::vector<double>      trformRotZZ_;
  std::vector<int>         layer_;
  std::vector<int>         layerIndex_;
  std::vector<int>         layerGroup_;
  std::vector<int>         cellFactor_; 
  std::vector<int>         depth_;
  std::vector<int>         depthIndex_;
  std::vector<int>         depthLayerF_;
  std::vector<double>      zLayerHex_;
  std::vector<double>      rMinLayHex_;
  std::vector<double>      rMaxLayHex_;
  std::vector<int>         waferCopy_;
  std::vector<int>         waferTypeL_;
  std::vector<int>         waferTypeT_;
  std::vector<double>      waferPosX_;
  std::vector<double>      waferPosY_;
  std::vector<double>      cellFineX_;
  std::vector<double>      cellFineY_;
  std::vector<bool>        cellFineHalf_;
  std::vector<double>      cellCoarseX_;
  std::vector<double>      cellCoarseY_;
  std::vector<bool>        cellCoarseHalf_;
  std::vector<int>         layerGroupM_;
  std::vector<int>         layerGroupO_;
  std::vector<double>      boundR_;
  std::vector<double>      rLimit_;
  std::vector<int>         cellFine_;
  std::vector<int>         cellCoarse_;
  double                   waferR_;
  int                      levelT_;
  int                      mode_;
  double                   slopeMin_;
  layer_map                copiesInLayers_;

  COND_SERIALIZABLE;
};

#endif
