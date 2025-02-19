#ifndef MuonDTDetLayerGeometryBuilder_h
#define MuonDTDetLayerGeometryBuilder_h

/** \class MuonDTDetLayerGeometryBuilder
 *
 *  Build the DT DetLayers.
 *
 *  $Date: 2006/06/02 12:21:39 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <vector>

class DetLayer;

class MuonDTDetLayerGeometryBuilder {
    public:
        /// Constructor
        MuonDTDetLayerGeometryBuilder();

        /// Destructor
        virtual ~MuonDTDetLayerGeometryBuilder();
  
        /// Operations
        static std::vector<DetLayer*> buildLayers(const DTGeometry& geo);
    private:
    
};
#endif

