#ifndef CondFormats_GeometryObjects_PHGCalParameters_h
#define CondFormats_GeometryObjects_PHGCalParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <string>
#include <vector>
#include <unordered_map>

class PHGCalParameters {

public:

  PHGCalParameters( void ) {}
  ~PHGCalParameters( void ) {}

  std::string              name_;
  std::vector<double>      cellSize_;
  std::vector<double>      moduleBlS_;
  std::vector<double>      moduleTlS_;
  std::vector<double>      moduleHS_;
  std::vector<double>      moduleDzS_;
  std::vector<double>      moduleAlphaS_;
  std::vector<double>      moduleCellS_;
  std::vector<double>      moduleBlR_;
  std::vector<double>      moduleTlR_;
  std::vector<double>      moduleHR_;
  std::vector<double>      moduleDzR_;
  std::vector<double>      moduleAlphaR_;
  std::vector<double>      moduleCellR_;
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
  std::vector<double>      zLayerHex_;
  std::vector<double>      rMinLayHex_;
  std::vector<double>      rMaxLayHex_;
  std::vector<double>      waferPosX_;
  std::vector<double>      waferPosY_;
  std::vector<double>      cellFineX_;
  std::vector<double>      cellFineY_;
  std::vector<double>      cellCoarseX_;
  std::vector<double>      cellCoarseY_;
  std::vector<double>      boundR_;
  std::vector<int>         moduleLayS_;
  std::vector<int>         moduleLayR_;
  std::vector<int>         layer_;
  std::vector<int>         layerIndex_;
  std::vector<int>         layerGroup_;
  std::vector<int>         cellFactor_; 
  std::vector<int>         depth_;
  std::vector<int>         depthIndex_;
  std::vector<int>         depthLayerF_;
  std::vector<int>         waferCopy_;
  std::vector<int>         waferTypeL_;
  std::vector<int>         waferTypeT_;
  std::vector<int>         layerGroupM_;
  std::vector<int>         layerGroupO_;
  std::vector<uint32_t>    trformIndex_;
  double                   waferR_;
  std::vector<double>      slopeMin_;
  int                      nCells_;
  int                      nSectors_;
  int                      mode_;
  std::vector< std::unordered_map<uint32_t, uint32_t> > copiesInLayers_;
  
  COND_SERIALIZABLE;
};

#endif
