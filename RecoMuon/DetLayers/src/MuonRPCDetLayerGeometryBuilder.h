#ifndef MuonRPCDetLayerGeometryBuilder_h
#define MuonRPCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: 2006/04/28 11:53:12 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>

class MuonRPCDetLayerGeometryBuilder {
    public:
        /// Constructor
        MuonRPCDetLayerGeometryBuilder();

        /// Destructor
        virtual ~MuonRPCDetLayerGeometryBuilder();
  
        /// Operations
        static pair<vector<DetLayer*>, vector<DetLayer*> > buildLayers(const RPCGeometry& geo);
    private:
    
};
#endif

