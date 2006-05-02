#ifndef MuonCSCDetLayerGeometryBuilder_h
#define MuonCSCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: 2006/04/28 11:53:12 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

class MuonCSCDetLayerGeometryBuilder {
    public:
        /// Constructor
        MuonCSCDetLayerGeometryBuilder();

        /// Destructor
        virtual ~MuonCSCDetLayerGeometryBuilder();
  
        /// Operations
        static pair<vector<DetLayer*>, vector<DetLayer*> > buildLayers(const CSCGeometry& geo);
    private:
    
};
#endif

