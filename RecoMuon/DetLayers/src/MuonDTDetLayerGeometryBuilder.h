#ifndef MuonDTDetLayerGeometryBuilder_h
#define MuonDTDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: 2006/04/28 11:53:12 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>

class MuonDTDetLayerGeometryBuilder {
    public:
        /// Constructor
        MuonDTDetLayerGeometryBuilder();

        /// Destructor
        virtual ~MuonDTDetLayerGeometryBuilder();
  
        /// Operations
        static pair<vector<DetLayer*>, vector<DetLayer*> > buildLayers(const DTGeometry& geo);
    private:
    
};
#endif

