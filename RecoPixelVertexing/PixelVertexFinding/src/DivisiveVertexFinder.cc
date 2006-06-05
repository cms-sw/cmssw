#include "RecoPixelVertexing/PixelVertexFinding/interface/DivisiveVertexFinder.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/PVPositionBuilder.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"

DivisiveVertexFinder::DivisiveVertexFinder(double zOffset, int ntrkMin, 
					   bool useError, double zSeparation, bool wtAverage)
  : zOffset_(zOffset), zSeparation_(zSeparation), ntrkMin_(ntrkMin), useError_(useError),
    wtAverage_(wtAverage)
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
  for (int i=0; i<trks.size(); i++) {
    v.add(trks[i]);
  }

  vertexes.push_back(v);

  return true;
}
