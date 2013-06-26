#ifndef PixelVertexFinding_PVCluster_h
#define PixelVertexFinding_PVCluster_h
/** \class PVCluster PVCluster.h RecoPixelVertexing/PixelVertexFinding/PVCluster.h 
 * A simple collection of tracks that represents a physical clustering
 * of charged particles, ie a vertex, in one dimension.  This
 * (typedef) class is used by the Pixel standalone vertex finding
 * classes found in RecoPixelVertexing/PixelVertexFinding.
 *
 *  $Date: 2006/05/09 09:52:44 $
 *  $Revision: 1.2 $
 *  \author Aaron Dominguez (UNL)
 */
#include "CommonTools/Clustering1D/interface/Cluster1D.h"
#include "DataFormats/TrackReco/interface/Track.h"

typedef Cluster1D<reco::Track> PVCluster;

#endif
