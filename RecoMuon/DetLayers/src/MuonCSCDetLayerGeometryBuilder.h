#ifndef MuonCSCDetLayerGeometryBuilder_h
#define MuonCSCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: 2006/04/25 14:01:18 $
 *  $Revision: 1.1 $
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
        static vector<MuRingForwardLayer*> buildLayers(const CSCGeometry& geo);

    private:
    
};
#endif

