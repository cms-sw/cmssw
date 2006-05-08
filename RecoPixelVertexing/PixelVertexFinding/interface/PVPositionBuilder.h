#ifndef RecoPixelVertexing_PVPositionBuilder_h
#define RecoPixelVertexing_PVPositionBuilder_h
/** \class PVPositionBuilder PVPositionBuilder.h RecoPixelVertexing/PixelVertexFinding/PVPositionBuilder.h 
 * This helper class calculates the average Z position of a collection of
 * tracks.  You have the option of calculating the straight average,
 * or making a weighted average using the error of the Z of the tracks.  This
 * class is used by the pixel vertexing to make a PVCluster and is
 * used by other PVCluster-related classes
 *
 *  $Date: 2006/05/03 20:05:00 $
 *  $Revision: 1.1 $
 *  \author Aaron Dominguez (UNL)
 */
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <vector>

class PVPositionBuilder {
 public:
  /// Constructor does nothing, no data members
  PVPositionBuilder();
   
  /// Calculate unweighted average of Z of tracks
  Measurement1D average(std::vector< const reco::Track* > & trks) const;

  /// Calculate Error-Weighted average of Z of tracks
  Measurement1D wtAverage(std::vector< const reco::Track* > & trks) const;
};
#endif
