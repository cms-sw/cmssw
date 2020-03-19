#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/BTauReco/interface/SeedingTrackFeatures.h"
#include "DataFormats/BTauReco/interface/TrackPairFeatures.h"

#include "RecoBTag/FeatureTools/interface/TrackPairInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/SeedingTrackInfoBuilder.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"

#include "RecoBTag/FeatureTools/interface/SeedingTracksConverter.h"

namespace btagbtvdeep {

  void seedingTracksToFeatures(const std::vector<reco::TransientTrack>& selectedTracks,
                               const std::vector<float>& masses,
                               const reco::Jet& jet,
                               const reco::Vertex& pv,
                               HistogramProbabilityEstimator* probabilityEstimator,
                               bool computeProbabilities,
                               std::vector<btagbtvdeep::SeedingTrackFeatures>& seedingT_features_vector)

  {
    GlobalVector jetdirection(jet.px(), jet.py(), jet.pz());
    GlobalPoint pvp(pv.x(), pv.y(), pv.z());

    std::multimap<double, std::pair<btagbtvdeep::SeedingTrackInfoBuilder, std::vector<btagbtvdeep::TrackPairFeatures>>>
        sortedSeedsMap;
    std::multimap<double, btagbtvdeep::TrackPairInfoBuilder> sortedNeighboursMap;

    std::vector<btagbtvdeep::TrackPairFeatures> tp_features_vector;

    sortedSeedsMap.clear();
    seedingT_features_vector.clear();

    std::vector<std::pair<bool, Measurement1D>> absIP3D(selectedTracks.size());
    std::vector<std::pair<bool, Measurement1D>> absIP2D(selectedTracks.size());
    std::vector<bool> absIP3D_filled(selectedTracks.size(), false);
    std::vector<bool> absIP2D_filled(selectedTracks.size(), false);

    unsigned int selTrackCount = 0;

    for (auto const& it : selectedTracks) {
      selTrackCount += 1;
      sortedNeighboursMap.clear();
      tp_features_vector.clear();

      if (reco::deltaR(it.track(), jet) > 0.4)
        continue;

      std::pair<bool, Measurement1D> ip = IPTools::absoluteImpactParameter3D(it, pv);

      absIP3D[selTrackCount - 1] = ip;
      absIP3D_filled[selTrackCount - 1] = true;

      std::pair<double, Measurement1D> jet_dist = IPTools::jetTrackDistance(it, jetdirection, pv);
      TrajectoryStateOnSurface closest =
          IPTools::closestApproachToJet(it.impactPointState(), pv, jetdirection, it.field());
      float length = 999;
      if (closest.isValid())
        length = (closest.globalPosition() - pvp).mag();

      if (!(ip.first && ip.second.value() >= 0.0 && ip.second.significance() >= 1.0 && ip.second.value() <= 9999. &&
            ip.second.significance() <= 9999. && it.track().normalizedChi2() < 5. &&
            std::fabs(it.track().dxy(pv.position())) < 2 && std::fabs(it.track().dz(pv.position())) < 17 &&
            jet_dist.second.value() < 0.07 && length < 5.))
        continue;

      std::pair<bool, Measurement1D> ip2d = IPTools::absoluteTransverseImpactParameter(it, pv);

      absIP2D[selTrackCount - 1] = ip2d;
      absIP2D_filled[selTrackCount - 1] = true;

      btagbtvdeep::SeedingTrackInfoBuilder seedInfo;
      seedInfo.buildSeedingTrackInfo(&(it),
                                     pv,
                                     jet,
                                     masses[selTrackCount - 1],
                                     ip,
                                     ip2d,
                                     jet_dist.second.value(),
                                     length,
                                     probabilityEstimator,
                                     computeProbabilities);

      unsigned int neighbourTrackCount = 0;

      for (auto const& tt : selectedTracks) {
        neighbourTrackCount += 1;

        if (neighbourTrackCount == selTrackCount)
          continue;
        if (std::fabs(pv.z() - tt.track().vz()) > 0.1)
          continue;

        //avoid calling IPs twice
        if (!absIP2D_filled[neighbourTrackCount - 1]) {
          absIP2D[neighbourTrackCount - 1] = IPTools::absoluteTransverseImpactParameter(tt, pv);
          absIP2D_filled[neighbourTrackCount - 1] = true;
        }

        if (!absIP3D_filled[neighbourTrackCount - 1]) {
          absIP3D[neighbourTrackCount - 1] = IPTools::absoluteImpactParameter3D(tt, pv);
          absIP3D_filled[neighbourTrackCount - 1] = true;
        }

        std::pair<bool, Measurement1D> t_ip = absIP3D[neighbourTrackCount - 1];
        std::pair<bool, Measurement1D> t_ip2d = absIP2D[neighbourTrackCount - 1];

        btagbtvdeep::TrackPairInfoBuilder trackPairInfo;
        trackPairInfo.buildTrackPairInfo(&(it), &(tt), pv, masses[neighbourTrackCount - 1], jetdirection, t_ip, t_ip2d);
        sortedNeighboursMap.insert(std::make_pair(trackPairInfo.pca_distance(), trackPairInfo));
      }

      int max_counter = 0;

      for (auto const& im : sortedNeighboursMap) {
        if (max_counter >= 20)
          break;
        btagbtvdeep::TrackPairFeatures tp_features;

        auto const& tp = im.second;

        tp_features.pt = (tp.track_pt() == 0) ? 0 : 1.0 / tp.track_pt();
        tp_features.eta = tp.track_eta();
        tp_features.phi = tp.track_phi();
        tp_features.mass = tp.track_candMass();
        tp_features.dz = logWithOffset(tp.track_dz());
        tp_features.dxy = logWithOffset(tp.track_dxy());
        tp_features.ip3D = log(tp.track_ip3d());
        tp_features.sip3D = log(tp.track_ip3dSig());
        tp_features.ip2D = log(tp.track_ip2d());
        tp_features.sip2D = log(tp.track_ip2dSig());
        tp_features.distPCA = log(tp.pca_distance());
        tp_features.dsigPCA = log(tp.pca_significance());
        tp_features.x_PCAonSeed = tp.pcaSeed_x();
        tp_features.y_PCAonSeed = tp.pcaSeed_y();
        tp_features.z_PCAonSeed = tp.pcaSeed_z();
        tp_features.xerr_PCAonSeed = tp.pcaSeed_xerr();
        tp_features.yerr_PCAonSeed = tp.pcaSeed_yerr();
        tp_features.zerr_PCAonSeed = tp.pcaSeed_zerr();
        tp_features.x_PCAonTrack = tp.pcaTrack_x();
        tp_features.y_PCAonTrack = tp.pcaTrack_y();
        tp_features.z_PCAonTrack = tp.pcaTrack_z();
        tp_features.xerr_PCAonTrack = tp.pcaTrack_xerr();
        tp_features.yerr_PCAonTrack = tp.pcaTrack_yerr();
        tp_features.zerr_PCAonTrack = tp.pcaTrack_zerr();
        tp_features.dotprodTrack = tp.dotprodTrack();
        tp_features.dotprodSeed = tp.dotprodSeed();
        tp_features.dotprodTrackSeed2D = tp.dotprodTrackSeed2D();
        tp_features.dotprodTrackSeed3D = tp.dotprodTrackSeed3D();
        tp_features.dotprodTrackSeedVectors2D = tp.dotprodTrackSeed2DV();
        tp_features.dotprodTrackSeedVectors3D = tp.dotprodTrackSeed3DV();
        tp_features.pvd_PCAonSeed = log(tp.pcaSeed_dist());
        tp_features.pvd_PCAonTrack = log(tp.pcaTrack_dist());
        tp_features.dist_PCAjetAxis = log(tp.pca_jetAxis_dist());
        tp_features.dotprod_PCAjetMomenta = tp.pca_jetAxis_dotprod();
        tp_features.deta_PCAjetDirs = log(tp.pca_jetAxis_dEta());
        tp_features.dphi_PCAjetDirs = tp.pca_jetAxis_dPhi();

        max_counter = max_counter + 1;
        tp_features_vector.push_back(tp_features);
      }

      sortedSeedsMap.insert(std::make_pair(-seedInfo.sip3d_Signed(), std::make_pair(seedInfo, tp_features_vector)));
    }

    int max_counter_seed = 0;

    for (auto const& im : sortedSeedsMap) {
      if (max_counter_seed >= 10)
        break;

      btagbtvdeep::SeedingTrackFeatures seed_features;

      auto const& seed = im.second.first;

      seed_features.nearTracks = im.second.second;
      seed_features.pt = (seed.pt() == 0) ? 0 : 1.0 / seed.pt();
      seed_features.eta = seed.eta();
      seed_features.phi = seed.phi();
      seed_features.mass = seed.mass();
      seed_features.dz = logWithOffset(seed.dz());
      seed_features.dxy = logWithOffset(seed.dxy());
      seed_features.ip3D = log(seed.ip3d());
      seed_features.sip3D = log(seed.sip3d());
      seed_features.ip2D = log(seed.ip2d());
      seed_features.sip2D = log(seed.sip2d());
      seed_features.signedIp3D = logWithOffset(seed.ip3d_Signed());
      seed_features.signedSip3D = logWithOffset(seed.sip3d_Signed());
      seed_features.signedIp2D = logWithOffset(seed.ip2d_Signed());
      seed_features.signedSip2D = logWithOffset(seed.sip2d_Signed());
      seed_features.trackProbability3D = seed.trackProbability3D();
      seed_features.trackProbability2D = seed.trackProbability2D();
      seed_features.chi2reduced = seed.chi2reduced();
      seed_features.nPixelHits = seed.nPixelHits();
      seed_features.nHits = seed.nHits();
      seed_features.jetAxisDistance = log(seed.jetAxisDistance());
      seed_features.jetAxisDlength = log(seed.jetAxisDlength());

      max_counter_seed = max_counter_seed + 1;
      seedingT_features_vector.push_back(seed_features);
    }

    if (sortedSeedsMap.size() < 10) {
      for (unsigned int i = sortedSeedsMap.size(); i < 10; i++) {
        std::vector<btagbtvdeep::TrackPairFeatures> tp_features_zeropad(20);
        btagbtvdeep::SeedingTrackFeatures seed_features_zeropad;
        seed_features_zeropad.nearTracks = tp_features_zeropad;
        seedingT_features_vector.push_back(seed_features_zeropad);
      }
    }
  }

}  // namespace btagbtvdeep
