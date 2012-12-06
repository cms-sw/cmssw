#include "RecoPixelVertexing/PixelVertexFinding/interface/DivisiveVertexFinder.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/PVPositionBuilder.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/PVCluster.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/PVClusterComparer.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

DivisiveVertexFinder::DivisiveVertexFinder(double zOffset, int ntrkMin, 
					   bool useError, double zSeparation, bool wtAverage,
					   int verbosity)
  : zOffset_(zOffset), zSeparation_(zSeparation), ntrkMin_(ntrkMin), useError_(useError),
    wtAverage_(wtAverage),
    divmeth_(zOffset, ntrkMin, useError, zSeparation, wtAverage),
    verbose_(verbosity)
{

}

DivisiveVertexFinder::~DivisiveVertexFinder(){}

bool DivisiveVertexFinder::findVertexes(const reco::TrackRefVector &trks,  // input
					reco::VertexCollection &vertexes){ // output
  PVPositionBuilder pos;
  Measurement1D vz;
  if (wtAverage_) {
    vz = pos.wtAverage(trks);
  }
  else {
    vz = pos.average(trks);
  }
  reco::Vertex::Error err;
  err(2,2) = vz.error()*vz.error();

  reco::Vertex v( reco::Vertex::Point(0,0,vz.value()), err, 0, 1, trks.size() );
    for (unsigned int i=0; i<trks.size(); i++) {
      double vz = trks[i]->vz();
      if(edm::isNotFinite(vz)) continue;
      v.add(reco::TrackBaseRef(trks[i]));
    }
    
  vertexes.push_back(v);

  return true;
}

bool DivisiveVertexFinder::findVertexesAlt(const reco::TrackRefVector &trks,  // input
					   reco::VertexCollection &vertexes, const math::XYZPoint & bs){ // output
  std::vector< PVCluster > in;
  std::pair< std::vector< PVCluster >, std::vector< const reco::Track* > > out;
  
  // Convert input track collection into container needed by Wolfgang's templated code
  // Need to save a map to reconvert from bare pointers, oy vey
  std::map< const reco::Track*, reco::TrackRef > mapa; 
  //  std::vector< std::vector< const reco::Track* > > trkps;
  for (unsigned int i=0; i<trks.size(); ++i) {
    double vz = trks[i]->vz();
    if(edm::isNotFinite(vz)) continue;
    std::vector< const reco::Track* > temp;
    temp.clear();
    temp.push_back( &(*trks[i]) );

    in.push_back( PVCluster( Measurement1D(trks[i]->dz(bs), trks[i]->dzError() ), temp ) );
    mapa[temp[0]] = trks[i];
  }

  if (verbose_ > 0 ) {
    edm::LogInfo("DivisiveVertexFinder") << "size of input vector of clusters " << in.size();
    for (unsigned int i=0; i<in.size(); ++i) {
      edm::LogInfo("DivisiveVertexFinder") << "Track " << i << " addr " << in[i].tracks()[0] 
					   << " dz " << in[i].tracks()[0]->dz(bs)
					   << " +- " << in[i].tracks()[0]->dzError()
					   << " prodID " << mapa[in[i].tracks()[0]].id()
					   << " dz from RefTrack " << mapa[in[i].tracks()[0]]->dz(bs)
					   << " +- " << mapa[in[i].tracks()[0]]->dzError();
    }
  }

  // Run the darn thing
  divmeth_.setBeamSpot(bs);
  out = divmeth_(in);

  if (verbose_ > 0) edm::LogInfo("DivisiveVertexFinder") << " DivisiveClusterizer1D found " 
							 << out.first.size() << " vertexes";

  // Now convert the output yet again into something we can safely store in the event
  for (unsigned int iv=0; iv<out.first.size(); ++iv) { // loop over output vertexes
    reco::Vertex::Error err;
    err(2,2) = out.first[iv].position().error()*out.first[iv].position().error();
    
    reco::Vertex v( reco::Vertex::Point(0,0,out.first[iv].position().value()), err, 0, 1, out.second.size() );
    if (verbose_ > 0 ) edm::LogInfo("DivisiveVertexFinder") << " DivisiveClusterizer1D vertex " << iv 
							    << " has " << out.first[iv].tracks().size()
							    << " tracks and a position of " << v.z() 
							    << " +- " << std::sqrt(v.covariance(2,2));
    for (unsigned int itrk=0; itrk<out.first[iv].tracks().size(); ++itrk) {
      v.add( reco::TrackBaseRef(mapa[out.first[iv].tracks()[itrk]] ) );
    }
    vertexes.push_back(v); // Done with horrible conversion, save it
  }

  // Finally, sort the vertexes in decreasing sumPtSquared
  std::sort(vertexes.begin(), vertexes.end(), PVClusterComparer());

  return true;
}
