#ifndef MuonCSCDetLayerGeometryBuilder_h
#define MuonCSCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: 2006/04/25 17:03:23 $
 *  $Revision: 1.2 $
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
        //static pair<vector<MuRingForwardLayer*>, vector<MuRingForwardLayer*> > buildLayers(const CSCGeometry& geo);
        static pair<vector<DetLayer*>, vector<DetLayer*> > buildLayers(const CSCGeometry& geo);
    private:
    
};
#endif

