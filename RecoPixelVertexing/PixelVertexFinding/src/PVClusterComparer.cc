#include "RecoPixelVertexing/PixelVertexFinding/interface/PVClusterComparer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TMath.h"

PVClusterComparer::PVClusterComparer()
  : track_pT_min_  ( 2.5)
  , track_pT_max_  (10.)
  , track_chi2_max_(9999999.)
  , track_prob_max_(9999999.)
{
}

PVClusterComparer::PVClusterComparer(double track_pt_min, double track_pt_max, double track_chi2_max, double track_prob_max)
  : track_pT_min_  (track_pt_min)
  , track_pT_max_  (track_pt_max)
  , track_chi2_max_(track_chi2_max)
  , track_prob_max_(track_prob_max)
{
}

double PVClusterComparer::pTSquaredSum(const PVCluster &v) const {
  double sum=0;
  for (unsigned int i=0; i<v.tracks().size(); ++i) {
    // RM : exclude badly reconstructed tracks from the sum
    if (TMath::Prob(v.tracks()[i]->chi2(),v.tracks()[i]->ndof())<track_prob_max_) continue ; 
    if (v.tracks()[i]->normalizedChi2()<track_chi2_max_) continue;
    double pt = v.tracks()[i]->pt();
    if (pt > 2.5) { // Don't count tracks below 2.5 GeV
      if (pt > track_pT_max_) pt = track_pT_max_;
      sum += pt*pt;
    }
  }
  return sum;
}
double PVClusterComparer::pTSquaredSum(const reco::Vertex &v) const {
  double sum=0;
  for (reco::Vertex::trackRef_iterator i=v.tracks_begin(); i!=v.tracks_end(); ++i) {
    // RM : exclude badly reconstructed tracks from the sum
    if (TMath::Prob((*i)->chi2(),(*i)->ndof())<track_prob_max_) continue ; 
    if ((*i)->normalizedChi2()<track_chi2_max_) continue;
    double pt = (*i)->pt();
    if (pt > 2.5) { // Don't count tracks below 2.5 GeV
      if (pt > track_pT_max_) pt = track_pT_max_;
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
