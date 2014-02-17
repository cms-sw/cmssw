#ifndef MuonRPCDetLayerGeometryBuilder_h
#define MuonRPCDetLayerGeometryBuilder_h

/** \class MuonRPCDetLayerGeometryBuilder
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
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include <vector>

class MuonRPCDetLayerGeometryBuilder {
 public:
  /// Constructor (disabled, only static access is allowed)
  MuonRPCDetLayerGeometryBuilder(){}

  /// Destructor
  virtual ~MuonRPCDetLayerGeometryBuilder();
  
  /// Builds the forward (+Z, return.first) and backward (-Z, return.second) layers.
  /// Both vectors are sorted inside-out
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildEndcapLayers(const RPCGeometry& geo);
        
  /// Builds the barrel layers. Result vector is sorted inside-out
  static std::vector<DetLayer*> buildBarrelLayers(const RPCGeometry& geo);
    
 private:
  static void makeBarrelLayers(std::vector<const GeomDet *> & geomDets,
                               std::vector<MuRodBarrelLayer*> & result);
  static void makeBarrelRods(std::vector<const GeomDet *> & geomDets,
                             std::vector<const DetRod*> & result);
  static bool isFront(const RPCDetId & rpcId);
  static MuRingForwardDoubleLayer* buildLayer(int endcap,std::vector<int> rings, int station,int layer, std::vector<int>& rolls,const RPCGeometry& geo);          
    
};
#endif

