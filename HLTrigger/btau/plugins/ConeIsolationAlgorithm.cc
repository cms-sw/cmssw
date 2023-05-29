#include "ConeIsolationAlgorithm.h"
using namespace std;
using namespace reco;
using namespace edm;

ConeIsolationAlgorithm::ConeIsolationAlgorithm() = default;

ConeIsolationAlgorithm::ConeIsolationAlgorithm(const ParameterSet& parameters) {
  //FIXME: use unsigned int where needed
  m_cutPixelHits = parameters.getParameter<int>("MinimumNumberOfPixelHits");  // not used
  m_cutTotalHits = parameters.getParameter<int>("MinimumNumberOfHits");
  m_cutMaxTIP = parameters.getParameter<double>("MaximumTransverseImpactParameter");
  m_cutMinPt = parameters.getParameter<double>("MinimumTransverseMomentum");
  m_cutMaxChiSquared = parameters.getParameter<double>("MaximumChiSquared");
  dZ_vertex = parameters.getParameter<double>("DeltaZetTrackVertex");  //  to be modified
  useVertexConstrain_ = parameters.getParameter<bool>("useVertex");

  matching_cone = parameters.getParameter<double>("MatchingCone");
  signal_cone = parameters.getParameter<double>("SignalCone");
  isolation_cone = parameters.getParameter<double>("IsolationCone");
  pt_min_isolation = parameters.getParameter<double>("MinimumTransverseMomentumInIsolationRing");
  pt_min_leadTrack = parameters.getParameter<double>("MinimumTransverseMomentumLeadingTrack");
  n_tracks_isolation_ring = parameters.getParameter<int>("MaximumNumberOfTracksIsolationRing");

  useFixedSizeCone = parameters.getParameter<bool>("UseFixedSizeCone");
  variableConeParameter = parameters.getParameter<double>("VariableConeParameter");
  variableMaxCone = parameters.getParameter<double>("VariableMaxCone");
  variableMinCone = parameters.getParameter<double>("VariableMinCone");
}

void ConeIsolationAlgorithm::fillDescription(edm::ParameterSetDescription& desc) {
  desc.add<int>("MinimumNumberOfPixelHits", 2);
  desc.add<int>("MinimumNumberOfHits", 8);
  desc.add<double>("MaximumTransverseImpactParameter", 0.03);
  desc.add<double>("MinimumTransverseMomentum", 1.0);
  desc.add<double>("MaximumChiSquared", 100.0);
  desc.add<double>("DeltaZetTrackVertex", 0.2);
  desc.add<bool>("useVertex", true);
  desc.add<double>("MatchingCone", 0.1);
  desc.add<double>("SignalCone", 0.07);
  desc.add<double>("IsolationCone", 0.45);
  desc.add<double>("MinimumTransverseMomentumInIsolationRing", 0.0);
  desc.add<double>("MinimumTransverseMomentumLeadingTrack", 6.0);
  desc.add<int>("MaximumNumberOfTracksIsolationRing", 0);
  desc.add<bool>("UseFixedSizeCone", true);
  desc.add<double>("VariableConeParameter", 3.5);
  desc.add<double>("VariableMaxCone", 0.17);
  desc.add<double>("VariableMinCone", 0.05);
}

pair<float, IsolatedTauTagInfo> ConeIsolationAlgorithm::tag(const JetTracksAssociationRef& jetTracks,
                                                            const Vertex& pv) const {
  const edm::RefVector<reco::TrackCollection>& tracks = jetTracks->second;
  edm::RefVector<reco::TrackCollection> myTracks;

  // Selection of the Tracks
  float z_pv = pv.z();
  for (auto&& track : tracks) {
    if ((track)->pt() > m_cutMinPt && (track)->normalizedChi2() < m_cutMaxChiSquared &&
        fabs((track)->dxy(pv.position())) < m_cutMaxTIP && (track)->recHitsSize() >= (unsigned int)m_cutTotalHits &&
        (track)->hitPattern().numberOfValidPixelHits() >= m_cutPixelHits) {
      if (useVertexConstrain_ && z_pv > -500.) {
        if (fabs((track)->dz(pv.position())) < dZ_vertex)
          myTracks.push_back(track);
      } else
        myTracks.push_back(track);
    }
  }
  IsolatedTauTagInfo resultExtended(myTracks, jetTracks);

  double r_sigCone = signal_cone;
  double energyJet = jetTracks->first->energy();
  if (not useFixedSizeCone) {
    r_sigCone = std::min(variableMaxCone, variableConeParameter / energyJet);
    r_sigCone = std::max((double)r_sigCone, variableMinCone);
  }

  // now I can use it for the discriminator;
  math::XYZVector jetDir(jetTracks->first->px(), jetTracks->first->py(), jetTracks->first->pz());
  float discriminator = 0.;
  if (useVertexConstrain_) {
    // In this case all the selected tracks comes from the same vertex, so no need to pass the dZ_vertex requirement to the discriminator
    const TrackRef myLeadTk = resultExtended.leadingSignalTrack(matching_cone, pt_min_leadTrack);
    resultExtended.setLeadingTrack(myLeadTk);
    discriminator = resultExtended.discriminator(
        jetDir, matching_cone, r_sigCone, isolation_cone, pt_min_leadTrack, pt_min_isolation, n_tracks_isolation_ring);
    resultExtended.setDiscriminator(discriminator);
  } else {
    // In this case the dZ_vertex is used to associate the tracks to the Z_imp parameter of the Leading Track
    const TrackRef myLeadTk = resultExtended.leadingSignalTrack(matching_cone, pt_min_leadTrack);
    resultExtended.setLeadingTrack(myLeadTk);
    discriminator = resultExtended.discriminator(jetDir,
                                                 matching_cone,
                                                 r_sigCone,
                                                 isolation_cone,
                                                 pt_min_leadTrack,
                                                 pt_min_isolation,
                                                 n_tracks_isolation_ring,
                                                 dZ_vertex);
    resultExtended.setDiscriminator(discriminator);
  }

  return std::make_pair(discriminator, resultExtended);
}
