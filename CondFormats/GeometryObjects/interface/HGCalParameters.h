#ifndef CondFormats_GeometryObjects_HGCalParameters_h
#define CondFormats_GeometryObjects_HGCalParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include<string>
#include<vector>
#include<iostream>
#include<CLHEP/Geometry/Transform3D.h>

class HGCalParameters {

public:
  
 HGCalParameters(const std::string& nam): name_(nam) { }
  ~HGCalParameters( void ) { }

  struct hgtrap {
    int           lay;
    float         bl, tl, h, dz, alpha, cellSize;
  };

  struct hgtrform {
    int                zp, lay, sec, subsec;
    bool               used;
    CLHEP::Hep3Vector  h3v;
    CLHEP::HepRotation hr;
  };

  std::string              name_;
  int                      nCells_;
  int                      nSectors_;
  std::vector<double>      cellSize_;
  std::vector<hgtrap>      modules_;
  std::vector<hgtrap>      moduler_;
  std::vector<hgtrform>    trform_;
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
  std::vector<GlobalPoint> waferPos_;
  std::vector<GlobalPoint> cellFine_;
  std::vector<GlobalPoint> cellCoarse_;
  std::vector<int>         layerGroupM_;
  std::vector<int>         layerGroupO_;
  std::vector<double>      boundR_;
  double                   waferR_;
  int                      mode_;

};

#endif
