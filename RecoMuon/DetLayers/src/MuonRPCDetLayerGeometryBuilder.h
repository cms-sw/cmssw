#ifndef MuonRPCDetLayerGeometryBuilder_h
#define MuonRPCDetLayerGeometryBuilder_h

/** \class MuonRPCDetLayerGeometryBuilder
 *
 *  Build the RPC DetLayers.
 *
 *  $Date: 2006/08/31 15:25:24 $
 *  $Revision: 1.6 $
 *  \author N. Amapane - CERN
 */

class DetLayer;
class MuRingForwardLayer;


#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
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
  static MuRingForwardLayer* buildLayer(int endcap,std::vector<int> rings, int station,int layer, std::vector<int>& rolls,const RPCGeometry& geo);          
    
};
#endif

