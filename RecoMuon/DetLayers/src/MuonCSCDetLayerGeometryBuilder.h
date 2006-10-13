#ifndef MuonCSCDetLayerGeometryBuilder_h
#define MuonCSCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  Build the CSC DetLayers.
 *
 *  $Date: 2006/06/02 12:21:39 $
 *  $Revision: 1.6 $
 *  \author N. Amapane - CERN
 */

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <vector>

class DetLayer;
class MuRingForwardLayer;

class MuonCSCDetLayerGeometryBuilder {
 public:

  /// return.first=forward (+Z), return.second=backward (-Z)
  /// both vectors are sorted inside-out
  static std::pair<std::vector<DetLayer*>, std::vector<DetLayer*> > buildLayers(const CSCGeometry& geo);
 private:
  // Disable constructor - only static access is allowed.
  MuonCSCDetLayerGeometryBuilder(){}

  static MuRingForwardLayer* buildLayer(int endcap,
					int station,
					std::vector<int>& rings,
					const CSCGeometry& geo);
};
#endif

