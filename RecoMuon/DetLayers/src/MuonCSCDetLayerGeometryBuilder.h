#ifndef MuonCSCDetLayerGeometryBuilder_h
#define MuonCSCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  Build the CSC DetLayers.
 *
 *  $Date: 2006/05/18 14:52:41 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - CERN
 */

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <vector>

class DetLayer;
class MuRingForwardLayer;

class MuonCSCDetLayerGeometryBuilder {
 public:
  /// Operations
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

