#ifndef MuonCSCDetLayerGeometryBuilder_h
#define MuonCSCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  Build the CSC DetLayers.
 *
 *  $Date: 2007/06/14 23:48:22 $
 *  $Revision: 1.8 $
 *  \author N. Amapane - CERN
 */

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <vector>

class DetLayer;
class MuRingForwardDoubleLayer;
class MuDetRing;

class MuonCSCDetLayerGeometryBuilder {
 public:

  /// return.first=forward (+Z), return.second=backward (-Z)
  /// both vectors are sorted inside-out
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildLayers(const CSCGeometry& geo);
 private:
  // Disable constructor - only static access is allowed.
  MuonCSCDetLayerGeometryBuilder(){}

  static MuRingForwardDoubleLayer* buildLayer(int endcap,
					int station,
					std::vector<int>& rings,
					const CSCGeometry& geo);

  static MuDetRing * makeDetRing(std::vector<const GeomDet*> & geomDets);
  static bool isFront(int station, int ring, int chamber);
};
#endif

