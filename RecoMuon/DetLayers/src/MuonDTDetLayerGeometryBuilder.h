#ifndef MuonDTDetLayerGeometryBuilder_h
#define MuonDTDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: 2006/05/02 10:23:02 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuRodBarrelLayer.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>

class MuonDTDetLayerGeometryBuilder {
    public:
        /// Constructor
        MuonDTDetLayerGeometryBuilder();

        /// Destructor
        virtual ~MuonDTDetLayerGeometryBuilder();
  
        /// Operations
        static vector<DetLayer*> buildLayers(const DTGeometry& geo);
    private:
    
};
#endif

