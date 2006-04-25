#ifndef MuonCSCDetLayerGeometryBuilder_h
#define MuonCSCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "RecoMuon/DetLayers/interface/MuRingForwardLayer.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

class MuonCSCDetLayerGeometryBuilder {
    public:
        /// Constructor
        MuonCSCDetLayerGeometryBuilder();

        /// Destructor
        virtual ~MuonCSCDetLayerGeometryBuilder();
  
        /// Operations
        vector<MuRingForwardLayer*> buildLayers(const CSCGeometry& geo);

    private:
    
};
#endif

