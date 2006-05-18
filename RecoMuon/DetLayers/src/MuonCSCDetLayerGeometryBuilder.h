#ifndef MuonCSCDetLayerGeometryBuilder_h
#define MuonCSCDetLayerGeometryBuilder_h

/** \class MuonCSCDetLayerGeometryBuilder
 *
 *  No description available.
 *
 *  $Date: 2006/05/02 10:35:28 $
 *  $Revision: 1.4 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <vector>

class MuonCSCDetLayerGeometryBuilder {
 public:
  /// Operations
  static pair<vector<DetLayer*>, vector<DetLayer*> > buildLayers(const CSCGeometry& geo);
 private:
  // Disable constructor - only static access is allowed.
  MuonCSCDetLayerGeometryBuilder(){}

  static MuRingForwardLayer* buildLayer(int endcap,
					int station,
					std::vector<int>& rings,
					const CSCGeometry& geo);
};
#endif

