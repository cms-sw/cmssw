#ifndef BTLDetLayerGeometryBuilder_h
#define BTLDetLayerGeometryBuilder_h

/** \class BTLDetLayerGeometryBuilder
 *
 *  Build the BTL DetLayers.
 *
 *  \author L. Gray - FNAL
 */

#include <Geometry/MTDGeometryBuilder/interface/MTDGeometry.h>
#include <vector>

class DetLayer;

class BTLDetLayerGeometryBuilder {
public:
  /// Constructor
  BTLDetLayerGeometryBuilder();

  /// Destructor
  virtual ~BTLDetLayerGeometryBuilder();

  /// Operations
  static std::vector<DetLayer*> buildLayers(const MTDGeometry& geo);

private:
};
#endif
