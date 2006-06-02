#ifndef MuonRPCDetLayerGeometryBuilder_h
#define MuonRPCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: 2006/05/02 10:22:56 $
 *  $Revision: 1.1 $
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
        
	static vector<DetLayer*> buildLayers(const RPCGeometry& geo);
    
    private:
    
};
#endif

