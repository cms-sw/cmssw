#ifndef RecoPixelVertexing_DivisiveVertexFinder_h
#define RecoPixelVertexing_DivisiveVertexFinder_h
/** \class DivisiveVertexFinder DivisiveVertexFinder.h RecoPixelVertexing/PixelVertexFinding/interface/DivisiveVertexFinder.h 

 Description: Fits a primary vertex in 1D (z) using the "divisive method"

 Implementation: 
 This class was ported from ORCA by me (Aaron).  It was originally written by ...
 Find the PV candidates with a simple divisive method.
 Divide the luminosity region in several regions according 
 to the track distance and for each of them make a PVCluster. 
 Iteratively discard tracks and recover them in a new PVCluster.
 Return a sorted vector<Vertex> (aka VertexCollection) with the z coordinate of PV candidates
 \param ntkmin Minimum number of tracks required to form a cluster.
 \param useError physical distances or weighted distances.
 \param zsep Maximum distance between two adjacent tracks that belong
 to the same initial cluster.
 \param wei Compute the cluster "center" with an unweighted or a weighted
 average of the tracks. Weighted means weighted with the error
 of the data point.

 $Date: 2006/05/23 14:10:00 $
 $Revision: 1.1 $
 \author Aaron Dominguez (UNL)
*/
#include <vector>
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class DivisiveVertexFinder {
 public:
  DivisiveVertexFinder(double zOffset=5.0, int ntrkMin=5, bool useError=true, 
		       double zSeparation=0.05, bool wtAverage=true);
  ~DivisiveVertexFinder();
  
  /// Run the divisive algorithm and return a vector of vertexes for the input track collection
  bool findVertexes(const std::vector< reco::TrackRef > &trks, // input
		    reco::VertexCollection &vertexes);         // output
 private:
  double zOffset_, zSeparation_;
  int ntrkMin_;
  bool useError_, wtAverage_;
};
#endif
