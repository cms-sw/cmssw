#include "DataFormats/BTauReco/interface/TauMassTagInfo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/RefVector.h"

using namespace edm;
using namespace reco;
using namespace std;

float reco::TauMassTagInfo::discriminator(
    double matching_cone, double leading_trk_pt, double signal_cone, double cluster_track_cone, double m_cut) const {
  float discriminator = 0.0;
  double invariantMass = getInvariantMass(matching_cone, leading_trk_pt, signal_cone, cluster_track_cone);
  if (invariantMass >= 0.0 && invariantMass < m_cut)
    discriminator = 1.0;
  return discriminator;
}
//
// -- Set IsolatedTauTag
//
void reco::TauMassTagInfo::setIsolatedTauTag(const IsolatedTauTagInfoRef isolationRef) { isolatedTau = isolationRef; }
//
// -- Get IsolatedTauTag
//
const IsolatedTauTagInfoRef& reco::TauMassTagInfo::getIsolatedTauTag() const { return isolatedTau; }
//
// -- Set Cluster Collection
//
void reco::TauMassTagInfo::storeClusterTrackCollection(reco::BasicClusterRef clusterRef, float dr) {
  clusterMap.insert(clusterRef, dr);
}
//
// -- Calculate 4 momentum vector from tracks only
//
bool reco::TauMassTagInfo::calculateTrkP4(double matching_cone,
                                          double leading_trk_pt,
                                          double signal_cone,
                                          math::XYZTLorentzVector& p4) const {
  const TrackRef leadTk = isolatedTau->leadingSignalTrack(matching_cone, leading_trk_pt);
  if (!leadTk) {
    std::cout << " TauMassTagInfo::  No Leading Track !!  " << std::endl;
    return false;
  }
  math::XYZVector momentum = (*leadTk).momentum();
  const RefVector<TrackCollection> signalTracks = isolatedTau->tracksInCone(momentum, signal_cone, 1.0);
  //  if (signalTracks.size() == 0 || signalTracks.size()%2 == 0) return false;
  if (signalTracks.empty())
    return false;

  double px_inv = 0.0;
  double py_inv = 0.0;
  double pz_inv = 0.0;
  double e_inv = 0.0;
  for (RefVector<TrackCollection>::const_iterator itrack = signalTracks.begin(); itrack != signalTracks.end();
       itrack++) {
    double p = (*itrack)->p();
    double energy = sqrt(p * p + 0.139 * 0.139);  // use correct value!
    px_inv += (*itrack)->px();
    py_inv += (*itrack)->py();
    pz_inv += (*itrack)->pz();
    e_inv += energy;
  }

  p4.SetPx(px_inv);
  p4.SetPy(py_inv);
  p4.SetPz(pz_inv);
  p4.SetE(e_inv);

  return true;
}
//
// -- Get Invariant Mass
//
double reco::TauMassTagInfo::getInvariantMassTrk(double matching_cone,
                                                 double leading_trk_pt,
                                                 double signal_cone) const {
  math::XYZTLorentzVector totalP4;
  if (!calculateTrkP4(matching_cone, leading_trk_pt, signal_cone, totalP4))
    return -1.0;
  return totalP4.M();
}
//
// -- Get Invariant Mass
//
double reco::TauMassTagInfo::getInvariantMass(double matching_cone,
                                              double leading_trk_pt,
                                              double signal_cone,
                                              double track_cone) const {
  math::XYZTLorentzVector totalP4;
  if (!calculateTrkP4(matching_cone, leading_trk_pt, signal_cone, totalP4))
    return -1.0;

  // Add Clusters away from tracks
  for (ClusterTrackAssociationCollection::const_iterator mapIter = clusterMap.begin(); mapIter != clusterMap.end();
       mapIter++) {
    const reco::BasicClusterRef& iclus = mapIter->key;
    float dr = mapIter->val;
    if (dr > track_cone) {
      math::XYZVector clus3Vec(iclus->x(), iclus->y(), iclus->z());
      double e = iclus->energy();
      double theta = clus3Vec.theta();
      double phi = clus3Vec.phi();
      double px = e * sin(theta) * cos(phi);
      double py = e * sin(theta) * sin(phi);
      double pz = e * cos(theta);
      math::XYZTLorentzVector p4(px, py, pz, e);
      totalP4 += p4;
    }
  }
  return totalP4.M();
}
