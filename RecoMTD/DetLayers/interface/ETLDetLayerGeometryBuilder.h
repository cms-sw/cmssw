#ifndef ETLDetLayerGeometryBuilder_h
#define ETLDetLayerGeometryBuilder_h

/** \class ETLDetLayerGeometryBuilder
 *
 *  Build the ETL DetLayers.
 *
 *  \author L. Gray - FNAL
 */

#include <Geometry/MTDGeometryBuilder/interface/MTDGeometry.h>
#include <Geometry/MTDNumberingBuilder/interface/MTDTopology.h>
#include <vector>

class DetLayer;
class MTDSectorForwardDoubleLayer;
class MTDDetSector;

class ETLDetLayerGeometryBuilder {
public:
  /// return.first=forward (+Z), return.second=backward (-Z)
  /// both vectors are sorted inside-out
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildLayers(const MTDGeometry& geo,
                                                                                const MTDTopology& topo);

private:
  // Disable constructor - only static access is allowed.
  ETLDetLayerGeometryBuilder() {}

  static MTDSectorForwardDoubleLayer* buildLayer(
      int endcap, int layer, std::vector<unsigned>& sectors, const MTDGeometry& geo, const MTDTopology& topo);

  static bool isFront(int layer, int ring, int module);
  static MTDDetSector* makeDetSector(std::vector<const GeomDet*>& geomDets, const MTDTopology& topo);
};
#endif
