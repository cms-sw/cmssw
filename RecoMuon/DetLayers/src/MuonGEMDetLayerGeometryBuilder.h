#ifndef MuonGEMDetLayerGeometryBuilder_h
#define MuonGEMDetLayerGeometryBuilder_h

/** \class MuonGEMDetLayerGeometryBuilder
 *
 *  Build the GEM DetLayers.
 *
 *  \author R. Radogna
 */

class DetLayer;
class MuRingForwardDoubleLayer;
//class MuRodBarrelLayer;
class MuDetRing;


#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include <vector>

class MuonGEMDetLayerGeometryBuilder {
 public:
  /// Constructor (disabled, only static access is allowed)
  MuonGEMDetLayerGeometryBuilder(){}

  /// Destructor
  virtual ~MuonGEMDetLayerGeometryBuilder();
  
  /// Builds the forward (+Z, return.first) and backward (-Z, return.second) layers.
  /// Both vectors are sorted inside-out
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildEndcapLayers(const GEMGeometry& geo);
    
 private:
  static MuRingForwardDoubleLayer* buildLayer(int endcap,std::vector<int>& rings, int station,int layer,std::vector<int>& chambers,std::vector<int>& rolls,const GEMGeometry& geo);          
  static bool isFront(const GEMDetId & gemId);
  static MuDetRing * makeDetRing(std::vector<const GeomDet*> & geomDets);
  
};
#endif

