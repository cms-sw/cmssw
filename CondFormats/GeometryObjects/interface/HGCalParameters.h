#ifndef CondFormats_GeometryObjects_HGCalParameters_h
#define CondFormats_GeometryObjects_HGCalParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include<string>
#include<vector>
#include<iostream>
#include<CLHEP/Geometry/Transform3D.h>

class HGCalParameters {

public:
  
 HGCalParameters(const std::string& nam): name_(nam) { }
  ~HGCalParameters( void ) { }

  struct hgtrap {
    hgtrap(int lay0, float bl0, float tl0, float h0, float dz0, float alpha0): 
      lay(lay0),bl(bl0),tl(tl0),h(h0),dz(dz0),alpha(alpha0),cellSize(0) {}
    int           lay;
    float         bl, tl, h, dz, alpha, cellSize;
  };

  struct hgtrform {
    hgtrform(int zp0, int lay0, int sec0, int subsec0): zp(zp0), lay(lay0), sec(sec0), subsec(subsec0),used(false) {}
    int                zp, lay, sec, subsec;
    bool               used;
    CLHEP::Hep3Vector  h3v;
    CLHEP::HepRotation hr;
  };

  std::string           name_;
  int                   nCells_;
  int                   nSectors_;
  int                   nLayers_;
  std::vector<double>   cellSize_;
  std::vector<hgtrap>   modules_;
  std::vector<hgtrap>   moduler_;
  std::vector<hgtrform> trform_;
  std::vector<int>      layer_;
  std::vector<int>      layerIndex_;
  std::vector<int>      layerGroup_;
  std::vector<int>      cellFactor_; 
  std::vector<int>      depth_;
  std::vector<int>      depthIndex_;
  int                   mode_;

  COND_SERIALIZABLE;
};

#endif
