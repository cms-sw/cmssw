#ifndef ETLDetLayerGeometryBuilder_h
#define ETLDetLayerGeometryBuilder_h

/** \class ETLDetLayerGeometryBuilder
 *
 *  Build the ETL DetLayers.
 *
 *  \author L. Gray - FNAL
 */

#include <Geometry/MTDGeometryBuilder/interface/MTDGeometry.h>
#include <vector>

class DetLayer;
class MTDRingForwardDoubleLayer;
class MTDDetRing;

class ETLDetLayerGeometryBuilder {
public:
  /// return.first=forward (+Z), return.second=backward (-Z)
  /// both vectors are sorted inside-out
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildLayers(const MTDGeometry& geo);

private:
  // Disable constructor - only static access is allowed.
  ETLDetLayerGeometryBuilder() {}

  static MTDRingForwardDoubleLayer* buildLayer(int endcap,
                                               int layer,
                                               std::vector<unsigned>& rings,
                                               const MTDGeometry& geo);

  static MTDDetRing* makeDetRing(std::vector<const GeomDet*>& geomDets);
  static bool isFront(int layer, int ring, int module);
};
#endif
