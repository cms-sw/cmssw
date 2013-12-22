#include "RecoPixelVertexing/PixelVertexFinding/interface/PVClusterComparer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

PVClusterComparer::PVClusterComparer(){}

double PVClusterComparer::pTSquaredSum(const PVCluster &v) const {
  double sum=0;
  for (unsigned int i=0; i<v.tracks().size(); ++i) {
    double pt = v.tracks()[i]->pt();
    if (pt > 2.5) { // Don't count tracks below 2.5 GeV
      if (pt > 10.0) pt = 10.0;
      sum += pt*pt;
    }
  }
  return sum;
}
double PVClusterComparer::pTSquaredSum(const reco::Vertex &v) const {
  double sum=0;
  for (reco::Vertex::trackRef_iterator i=v.tracks_begin(); i!=v.tracks_end(); ++i) {
    double pt = (*i)->pt();
    if (pt > 2.5) { // Don't count tracks below 2.5 GeV
      if (pt > 10.0) pt = 10.0;
      sum += pt*pt;
    }
  }
  return sum;
}

bool PVClusterComparer::operator() (const PVCluster &v1, const PVCluster &v2) const {
  return ( pTSquaredSum(v1) > pTSquaredSum(v2) );
}
bool PVClusterComparer::operator() (const reco::Vertex &v1, const reco::Vertex &v2) const {
  return ( pTSquaredSum(v1) > pTSquaredSum(v2) );
}
