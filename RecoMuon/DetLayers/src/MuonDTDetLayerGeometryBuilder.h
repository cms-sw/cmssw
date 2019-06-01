#ifndef MuonDTDetLayerGeometryBuilder_h
#define MuonDTDetLayerGeometryBuilder_h

/** \class MuonDTDetLayerGeometryBuilder
 *
 *  Build the DT DetLayers.
 *
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
