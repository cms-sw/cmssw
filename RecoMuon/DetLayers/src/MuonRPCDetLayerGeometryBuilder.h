#ifndef MuonRPCDetLayerGeometryBuilder_h
#define MuonRPCDetLayerGeometryBuilder_h

/** \class MuonRPCDetLayerGeometryBuilder
 *
 *  Build the RPC DetLayers.
 *
 *  $Date: 2006/07/31 21:45:49 $
 *  $Revision: 1.5 $
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
  
  /// Builds the forward (first) and backward (second) layers
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildEndcapLayers(const RPCGeometry& geo);
        
  /// Builds the barrel layers
  static std::vector<DetLayer*> buildBarrelLayers(const RPCGeometry& geo);
    
 private:
  static MuRingForwardLayer* buildLayer(int endcap,std::vector<int> rings, int station,int layer, std::vector<int>& rolls,const RPCGeometry& geo);          
    
};
#endif

