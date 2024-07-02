#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"
#include "TrackstersPCA.h"

#include <iostream>
#include <set>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <functional>

void ticl::assignPCAtoTracksters(std::vector<Trackster> &tracksters,
                                 const std::vector<reco::CaloCluster> &layerClusters,
                                 const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                 double z_limit_em,
                                 const hgcal::RecHitTools &rhtools,
                                 bool computeLocalTime,
                                 bool energyWeight,
                                 bool clean,
                                 int minLayer,
                                 int maxLayer) {
  LogDebug("TrackstersPCA_Eigen") << "------- Eigen -------" << std::endl;

  for (auto &trackster : tracksters) {
    LogDebug("TrackstersPCA_Eigen") << "start testing teackster with size:" << trackster.vertices().size() << std::endl;

    Eigen::Vector3f point;
    point << 0., 0., 0.;
    Eigen::Vector3f barycenter;
    barycenter << 0., 0., 0.;
    Eigen::Vector3f filtered_barycenter;
    filtered_barycenter << 0., 0., 0.;

    auto fillPoint = [&](const reco::CaloCluster &c, const float weight = 1.f) {
      point[0] = weight * c.x();
      point[1] = weight * c.y();
      point[2] = weight * c.z();
    };

    // Initialize this trackster with default, dummy values
    trackster.setRawEnergy(0.f);
    trackster.setRawEmEnergy(0.f);
    trackster.setRawPt(0.f);
    trackster.setRawEmPt(0.f);

    size_t N = trackster.vertices().size();  ///< Number of layer clusters in trackster
    if (N == 0)
      continue;
    float weight = 1.f / N;
    float weights2_sum = 0.f;

    std::vector<float> layerClusterEnergies;

    for (size_t i = 0; i < N; ++i) {
      auto fraction = 1.f / trackster.vertex_multiplicity(i);
      trackster.addToRawEnergy(layerClusters[trackster.vertices(i)].energy() * fraction);
      if (std::abs(layerClusters[trackster.vertices(i)].z()) <= z_limit_em)
        trackster.addToRawEmEnergy(layerClusters[trackster.vertices(i)].energy() * fraction);

      // Compute the weighted barycenter.
      if (energyWeight)
        weight = layerClusters[trackster.vertices(i)].energy() * fraction;
      fillPoint(layerClusters[trackster.vertices(i)], weight);
      for (size_t j = 0; j < 3; ++j)
        barycenter[j] += point[j];

      layerClusterEnergies.push_back(layerClusters[trackster.vertices(i)].energy());
    }
    float raw_energy = trackster.raw_energy();
    float inv_raw_energy = 1.f / raw_energy;
    if (energyWeight)
      barycenter *= inv_raw_energy;
    trackster.setBarycenter(ticl::Trackster::Vector(barycenter));

    LogDebug("TrackstersPCA_Eigen") << "cleaning is  :" << clean << std::endl;

    std::vector<unsigned> filtered_idx;  // indices of layer clusters to consider for cleaned PCA
    float filtered_energy = 0;
    float inv_filtered_energy = 0;
    if (clean) {
      // Filter layerclusters for the cleaned PCA
      auto maxE_vertex = std::distance(layerClusterEnergies.begin(),
                                       std::max_element(layerClusterEnergies.begin(), layerClusterEnergies.end()));
      auto maxE_layer = getLayerFromLC(layerClusters[trackster.vertices(maxE_vertex)], rhtools);

      auto vertices_by_layer = sortByLayer(trackster, layerClusters, rhtools);

      for (unsigned i = 1; i <= rhtools.lastLayer(); ++i) {
        auto vertices_in_layer = vertices_by_layer[i];
        if (vertices_in_layer.empty())
          continue;

        std::vector<float> energies_in_layer;
        for (auto vrt : vertices_in_layer)
          energies_in_layer.push_back(layerClusters[trackster.vertices(vrt)].energy());

        unsigned maxEid_inLayer = std::distance(energies_in_layer.begin(),
                                                std::max_element(energies_in_layer.begin(), energies_in_layer.end()));

        // layer based filtering of what goes into the PCA
        if ((int)i >= (int)maxE_layer - minLayer && (int)i <= (int)maxE_layer + maxLayer) {
          auto filtered_vert = vertices_in_layer[maxEid_inLayer];
          filtered_idx.push_back(filtered_vert);

          const auto &maxE_LC = layerClusters[trackster.vertices(filtered_vert)];
          fillPoint(maxE_LC, maxE_LC.energy() * (1.f / trackster.vertex_multiplicity(filtered_vert)));
          for (size_t j = 0; j < 3; ++j)
            filtered_barycenter[j] += point[j];
          filtered_energy += maxE_LC.energy();
        }
      }
      inv_filtered_energy = 1. / filtered_energy;
      filtered_barycenter *= inv_filtered_energy;
    }
    LogDebug("TrackstersPCA_Eigen") << "min, max " << minLayer << "  " << maxLayer << std::endl;

    std::pair<float, float> timeTrackster;
    if (computeLocalTime)
      timeTrackster = ticl::computeLocalTracksterTime(trackster, layerClusters, layerClustersTime, barycenter, N);
    else
      timeTrackster = ticl::computeTracksterTime(trackster, layerClustersTime, N);

    trackster.setTimeAndError(timeTrackster.first, timeTrackster.second);
    LogDebug("TrackstersPCA") << "Use energy weighting: " << energyWeight << std::endl;
    LogDebug("TrackstersPCA") << "\nTrackster characteristics: " << std::endl;
    LogDebug("TrackstersPCA") << "Size: " << N << std::endl;
    LogDebug("TrackstersPCA") << "Energy: " << trackster.raw_energy() << std::endl;
    LogDebug("TrackstersPCA") << "raw_pt: " << trackster.raw_pt() << std::endl;
    LogDebug("TrackstersPCA") << "Means:          " << barycenter[0] << ", " << barycenter[1] << ", " << barycenter[2]
                              << std::endl;
    LogDebug("TrackstersPCA") << "Time:          " << trackster.time() << " +/- " << trackster.timeError() << std::endl;

    if (N > 2) {
      Eigen::Vector3f sigmas;
      sigmas << 0., 0., 0.;
      Eigen::Vector3f sigmasEigen;
      sigmasEigen << 0., 0., 0.;
      Eigen::Matrix3f covM = Eigen::Matrix3f::Zero();
      // Compute the Covariance Matrix and the sum of the squared weights, used
      // to compute the correct normalization.
      // The barycenter has to be known.

      auto calc_covM = [&](size_t i) {
        fillPoint(layerClusters[trackster.vertices(i)]);
        if (energyWeight && trackster.raw_energy()) {
          weight = (layerClusters[trackster.vertices(i)].energy() / trackster.vertex_multiplicity(i)) *
                   (clean ? inv_filtered_energy : inv_raw_energy);
          if (trackster.vertex_multiplicity(i) > 1)
            LogDebug("TrackstersPCA_Eigen")
                << "trackster.vertex_multiplicity(i)   :" << trackster.vertex_multiplicity(i);
        }
        weights2_sum += weight * weight;
        for (size_t x = 0; x < 3; ++x) {
          for (size_t y = 0; y <= x; ++y) {
            covM(x, y) += weight * (point[x] - (clean ? filtered_barycenter[x] : barycenter[x])) *
                          (point[y] - (clean ? filtered_barycenter[y] : barycenter[y]));
            covM(y, x) = covM(x, y);
          }
        }
      };
      if (clean) {
        for (size_t i : filtered_idx) {
          calc_covM(i);
        }
      } else {
        for (size_t i = 0; i < N; ++i) {
          calc_covM(i);
        }
      }

      covM *= 1.f / (1.f - weights2_sum);

      // Perform the actual decomposition
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>::RealVectorType eigenvalues_fromEigen;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>::EigenvectorsType eigenvectors_fromEigen;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covM);
      if (eigensolver.info() != Eigen::Success) {
        eigenvalues_fromEigen = eigenvalues_fromEigen.Zero();
        eigenvectors_fromEigen = eigenvectors_fromEigen.Zero();
      } else {
        eigenvalues_fromEigen = eigensolver.eigenvalues();
        eigenvectors_fromEigen = eigensolver.eigenvectors();
      }

      // Compute the spread in the both spaces.
      auto calc_spread = [&](size_t i) {
        fillPoint(layerClusters[trackster.vertices(i)]);
        sigmas += weight * (point - (clean ? filtered_barycenter : barycenter)).cwiseAbs2();
        Eigen::Vector3f point_transformed =
            eigenvectors_fromEigen * (point - (clean ? filtered_barycenter : barycenter));
        if (energyWeight && raw_energy)
          weight = (layerClusters[trackster.vertices(i)].energy() / trackster.vertex_multiplicity(i)) *
                   (clean ? inv_filtered_energy : inv_raw_energy);
        sigmasEigen += weight * (point_transformed.cwiseAbs2());
      };

      if (clean) {
        for (size_t i : filtered_idx) {
          calc_spread(i);
        }
      } else {
        for (size_t i = 0; i < N; ++i) {
          calc_spread(i);
        }
      }

      sigmas /= (1.f - weights2_sum);
      sigmasEigen /= (1.f - weights2_sum);

      trackster.fillPCAVariables(eigenvalues_fromEigen,
                                 eigenvectors_fromEigen,
                                 sigmas,
                                 sigmasEigen,
                                 3,
                                 ticl::Trackster::PCAOrdering::ascending);

      LogDebug("TrackstersPCA") << "EigenValues from Eigen/Tr(cov): " << eigenvalues_fromEigen[2] / covM.trace() << ", "
                                << eigenvalues_fromEigen[1] / covM.trace() << ", "
                                << eigenvalues_fromEigen[0] / covM.trace() << std::endl;
      LogDebug("TrackstersPCA") << "EigenValues from Eigen:         " << eigenvalues_fromEigen[2] << ", "
                                << eigenvalues_fromEigen[1] << ", " << eigenvalues_fromEigen[0] << std::endl;
      LogDebug("TrackstersPCA") << "EigenVector 3 from Eigen: " << eigenvectors_fromEigen(0, 2) << ", "
                                << eigenvectors_fromEigen(1, 2) << ", " << eigenvectors_fromEigen(2, 2) << std::endl;
      LogDebug("TrackstersPCA") << "EigenVector 2 from Eigen: " << eigenvectors_fromEigen(0, 1) << ", "
                                << eigenvectors_fromEigen(1, 1) << ", " << eigenvectors_fromEigen(2, 1) << std::endl;
      LogDebug("TrackstersPCA") << "EigenVector 1 from Eigen: " << eigenvectors_fromEigen(0, 0) << ", "
                                << eigenvectors_fromEigen(1, 0) << ", " << eigenvectors_fromEigen(2, 0) << std::endl;
      LogDebug("TrackstersPCA") << "Original sigmas:          " << sigmas[0] << ", " << sigmas[1] << ", " << sigmas[2]
                                << std::endl;
      LogDebug("TrackstersPCA") << "SigmasEigen in PCA space: " << sigmasEigen[2] << ", " << sigmasEigen[1] << ", "
                                << sigmasEigen[0] << std::endl;
      LogDebug("TrackstersPCA") << "covM:     \n" << covM << std::endl;
    }
  }
}

std::pair<float, float> ticl::computeLocalTracksterTime(const Trackster &trackster,
                                                        const std::vector<reco::CaloCluster> &layerClusters,
                                                        const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                                        const Eigen::Vector3f &barycenter,
                                                        size_t N) {
  float tracksterTime = 0.;
  float tracksterTimeErr = 0.;
  std::set<uint32_t> usedLC;

  auto project_lc_to_pca = [](const std::vector<double> &point, const std::vector<double> &segment_end) {
    double dot_product = 0.0;
    double segment_dot = 0.0;

    for (int i = 0; i < 3; ++i) {
      dot_product += point[i] * segment_end[i];
      segment_dot += segment_end[i] * segment_end[i];
    }

    double projection = 0.0;
    if (segment_dot != 0.0) {
      projection = dot_product / segment_dot;
    }

    std::vector<double> closest_point(3);
    for (int i = 0; i < 3; ++i) {
      closest_point[i] = projection * segment_end[i];
    }

    double distance = 0.0;
    for (int i = 0; i < 3; ++i) {
      distance += std::pow(point[i] - closest_point[i], 2);
    }

    return std::sqrt(distance);
  };

  constexpr double c = 29.9792458;  // cm/ns
  for (size_t i = 0; i < N; ++i) {
    // Add timing from layerClusters not already used
    if ((usedLC.insert(trackster.vertices(i))).second) {
      float timeE = layerClustersTime.get(trackster.vertices(i)).second;
      if (timeE > 0.f) {
        float time = layerClustersTime.get(trackster.vertices(i)).first;
        timeE = 1.f / pow(timeE, 2);
        float x = layerClusters[trackster.vertices(i)].x();
        float y = layerClusters[trackster.vertices(i)].y();
        float z = layerClusters[trackster.vertices(i)].z();

        if (project_lc_to_pca({x, y, z}, {barycenter[0], barycenter[1], barycenter[2]}) < 3) {  // set MR to 3
          float deltaT = 1.f / c *
                         std::sqrt(((barycenter[2] / z - 1.f) * x) * ((barycenter[2] / z - 1.f) * x) +
                                   ((barycenter[2] / z - 1.f) * y) * ((barycenter[2] / z - 1.f) * y) +
                                   (barycenter[2] - z) * (barycenter[2] - z));
          time = std::abs(barycenter[2]) < std::abs(z) ? time - deltaT : time + deltaT;

          tracksterTime += time * timeE;
          tracksterTimeErr += timeE;
        }
      }
    }
  }
  if (tracksterTimeErr > 0.f)
    return {tracksterTime / tracksterTimeErr, 1.f / std::sqrt(tracksterTimeErr)};
  else
    return {-99.f, -1.f};
}

std::pair<float, float> ticl::computeTracksterTime(const Trackster &trackster,
                                                   const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                                   size_t N) {
  std::vector<float> times;
  std::vector<float> timeErrors;
  std::set<uint32_t> usedLC;

  for (size_t i = 0; i < N; ++i) {
    // Add timing from layerClusters not already used
    if ((usedLC.insert(trackster.vertices(i))).second) {
      float timeE = layerClustersTime.get(trackster.vertices(i)).second;
      if (timeE > 0.f) {
        times.push_back(layerClustersTime.get(trackster.vertices(i)).first);
        timeErrors.push_back(1.f / pow(timeE, 2));
      }
    }
  }

  hgcalsimclustertime::ComputeClusterTime timeEstimator;
  return timeEstimator.fixSizeHighestDensity(times, timeErrors);
}
