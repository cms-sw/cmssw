
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/02/22 18:43:20 $
 *  $Revision: 1.1 $
 */

#include "CalibMuon/DTCalibration/interface/DTRecHitSegmentResidual.h"

//Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//RecHit
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

float DTRecHitSegmentResidual::compute(const DTGeometry* dtGeom, const DTRecHit1D& recHit1D, const DTRecSegment4D& segment) {

  const DTWireId wireId = recHit1D.wireId();
      
  // Get the layer and the wire position
  const DTLayer* layer = dtGeom->layer(wireId);
  float wireX = layer->specificTopology().wirePosition(wireId.wire());
      
  // Extrapolate the segment to the z of the wire
  // Get wire position in chamber RF
  // (y and z must be those of the hit to be coherent in the transf. of RF in case of rotations of the layer alignment)
  LocalPoint wirePosInLay(wireX,recHit1D.localPosition().y(),recHit1D.localPosition().z());
  GlobalPoint wirePosGlob = layer->toGlobal(wirePosInLay);
  const DTChamber* chamber = dtGeom->chamber(wireId.layerId().chamberId());
  LocalPoint wirePosInChamber = chamber->toLocal(wirePosGlob);
      
  // Segment position at Wire z in chamber local frame
  LocalPoint segPosAtZWire = segment.localPosition() + segment.localDirection()*wirePosInChamber.z()/cos(segment.localDirection().theta());
      
  // Compute the distance of the segment from the wire
  int sl = wireId.superlayer();
  float segmDistance = -1;
  if(sl == 1 || sl == 3) segmDistance = fabs(wirePosInChamber.x() - segPosAtZWire.x());
  else if(sl == 2) segmDistance =  fabs(segPosAtZWire.y() - wirePosInChamber.y());

  // Compute the distance of the recHit from the wire
  float recHitWireDist = fabs( recHit1D.localPosition().x() - wireX );
 
  // Compute the residuals 
  float residualOnDistance = recHitWireDist - segmDistance;

  return residualOnDistance; 
}
