#ifndef MuonGEMDetLayerGeometryBuilder_h
#define MuonGEMDetLayerGeometryBuilder_h

/** \class MuonGEMDetLayerGeometryBuilder
 *
 *  Build the RPC DetLayers.
 *
 *  $Date: 2007/07/08 04:20:26 $
 *  $Revision: 1.9 $
 *  \author N. Amapane - CERN
 */

class DetLayer;
class MuRingForwardDoubleLayer;
class MuRodBarrelLayer;


#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
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
  static bool isFront(const GEMDetId & gemId);

  static MuRingForwardDoubleLayer* buildLayer(int endcap,std::vector<int> rings, int station,int layer, std::vector<int>& rolls,const GEMGeometry& geo);          
  
};
#endif

