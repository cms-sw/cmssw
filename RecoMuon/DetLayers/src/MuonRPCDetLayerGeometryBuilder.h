#ifndef MuonRPCDetLayerGeometryBuilder_h
#define MuonRPCDetLayerGeometryBuilder_h

/** \class MuonRPCDetLayerGeometryBuilder
 *
 *  Build the RPC DetLayers.
 *
 *  $Date: 2006/06/02 12:21:39 $
 *  $Revision: 1.3 $
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
  
  /// Operations
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildEndcapLayers(const RPCGeometry& geo);
        
  static std::vector<DetLayer*> buildBarrelLayers(const RPCGeometry& geo);
    
 private:
  static MuRingForwardLayer* buildLayer(int endcap,int ring, int station,int layer, std::vector<int>& rolls,const RPCGeometry& geo);          
    
};
#endif

