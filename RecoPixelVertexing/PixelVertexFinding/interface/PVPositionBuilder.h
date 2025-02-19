#ifndef RecoPixelVertexing_PVPositionBuilder_h
#define RecoPixelVertexing_PVPositionBuilder_h
/** \class PVPositionBuilder PVPositionBuilder.h RecoPixelVertexing/PixelVertexFinding/PVPositionBuilder.h 
 * This helper class calculates the average Z position of a collection of
 * tracks.  You have the option of calculating the straight average,
 * or making a weighted average using the error of the Z of the tracks.  This
 * class is used by the pixel vertexing to make a PVCluster and is
 * used by other PVCluster-related classes
 *
 *  $Date: 2007/03/12 00:50:36 $
 *  $Revision: 1.4 $
 *  \author Aaron Dominguez (UNL)
 */
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <vector>

class PVPositionBuilder {
 public:
  /// Constructor does nothing, no data members
  PVPositionBuilder();
   
  /// Calculate unweighted average of Z of tracks from const collection of track pointers
  Measurement1D average(const reco::TrackRefVector& trks) const;

  /// Calculate Error-Weighted average of Z of tracks from const collection of track pointers
  Measurement1D wtAverage(const reco::TrackRefVector& trks) const;
};
#endif
