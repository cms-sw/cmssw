#ifndef MuonME0DetLayerGeometryBuilder_h
#define MuonME0DetLayerGeometryBuilder_h

/** \class MuonME0DetLayerGeometryBuilder
 *
 *  Build the ME0 DetLayers.
 *
 *  \author D. Nash
 */

class DetLayer;
//class MuRingForwardDoubleLayer;
class MuRingForwardLayer;
//class MuRodBarrelLayer;
class MuDetRing;


#include <Geometry/GEMGeometry/interface/ME0Geometry.h>
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include <vector>

class MuonME0DetLayerGeometryBuilder {
 public:
  /// Constructor (disabled, only static access is allowed)
  MuonME0DetLayerGeometryBuilder(){}

  /// Destructor
  virtual ~MuonME0DetLayerGeometryBuilder();
  
  /// Builds the forward (+Z, return.first) and backward (-Z, return.second) layers.
  /// Both vectors are sorted inside-out
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildEndcapLayers(const ME0Geometry& geo);
    
 private:
  //static MuRingForwardDoubleLayer* buildLayer(int endcap,int layer,std::vector<int>& chambers,std::vector<int>& rolls,const ME0Geometry& geo);          
  //static MuRingForwardLayer* buildLayer(int endcap,int layer,std::vector<int>& chambers,std::vector<int>& rolls,const ME0Geometry& geo);          
  static MuRingForwardLayer* buildLayer(int endcap, int layer, std::vector<int>& chambers,std::vector<int>& rolls,const ME0Geometry& geo);          
  static bool isFront(const ME0DetId & me0Id);
  static MuDetRing * makeDetRing(std::vector<const GeomDet*> & geomDets);
  
};
#endif

