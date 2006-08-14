#ifndef MuonRPCDetLayerGeometryBuilder_h
#define MuonRPCDetLayerGeometryBuilder_h

/** \class MuonRPCDetLayerGeometryBuilder
 *
 *  Build the RPC DetLayers.
 *
 *  $Date: 2006/06/02 08:46:32 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

class DetLayer;

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <vector>

class MuonRPCDetLayerGeometryBuilder {
 public:
  /// Constructor
  MuonRPCDetLayerGeometryBuilder();

  /// Destructor
  virtual ~MuonRPCDetLayerGeometryBuilder();
  
  /// Operations
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildEndcapLayers(const RPCGeometry& geo);
        
  static std::vector<DetLayer*> buildBarrelLayers(const RPCGeometry& geo);
    
 private:
    
};
#endif

