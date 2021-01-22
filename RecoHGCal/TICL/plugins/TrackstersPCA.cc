#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"
#include "TrackstersPCA.h"

#include <iostream>
#include <set>

#include <Eigen/Core>
#include <Eigen/Dense>

void ticl::assignPCAtoTracksters(std::vector<Trackster> &tracksters,
                                 const std::vector<reco::CaloCluster> &layerClusters,
                                 const edm::ValueMap<std::pair<float, float>> &layerClustersTime,
                                 double z_limit_em,
                                 bool energyWeight) {
  LogDebug("TrackstersPCA_Eigen") << "------- Eigen -------" << std::endl;

  for (auto &trackster : tracksters) {
    Eigen::Vector3d point;
    point << 0., 0., 0.;
    Eigen::Vector3d barycenter;
    barycenter << 0., 0., 0.;

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

    size_t N = trackster.vertices().size();
    float weight = 1.f / N;
    float weights2_sum = 0.f;
    Eigen::Vector3d sigmas;
    sigmas << 0., 0., 0.;
    Eigen::Vector3d sigmasEigen;
    sigmasEigen << 0., 0., 0.;
    Eigen::Matrix3d covM = Eigen::Matrix3d::Zero();

    std::vector<float> times;
    std::vector<float> timeErrors;
    std::set<uint32_t> usedLC;

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

      // Add timing from layerClusters not already used
      if ((usedLC.insert(trackster.vertices(i))).second) {
        float timeE = layerClustersTime.get(trackster.vertices(i)).second;
        if (timeE > 0.f) {
          times.push_back(layerClustersTime.get(trackster.vertices(i)).first);
          timeErrors.push_back(1. / pow(timeE, 2));
        }
      }
    }
    if (energyWeight && trackster.raw_energy())
      barycenter /= trackster.raw_energy();

    hgcalsimclustertime::ComputeClusterTime timeEstimator;
    std::pair<float, float> timeTrackster = timeEstimator.fixSizeHighestDensity(times, timeErrors);

    // Compute the Covariance Matrix and the sum of the squared weights, used
    // to compute the correct normalization.
    // The barycenter has to be known.
    for (size_t i = 0; i < N; ++i) {
      fillPoint(layerClusters[trackster.vertices(i)]);
      if (energyWeight && trackster.raw_energy())
        weight =
            (layerClusters[trackster.vertices(i)].energy() / trackster.vertex_multiplicity(i)) / trackster.raw_energy();
      weights2_sum += weight * weight;
      for (size_t x = 0; x < 3; ++x)
        for (size_t y = 0; y <= x; ++y) {
          covM(x, y) += weight * (point[x] - barycenter[x]) * (point[y] - barycenter[y]);
          covM(y, x) = covM(x, y);
        }
    }
    covM *= 1. / (1. - weights2_sum);

    // Perform the actual decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>::RealVectorType eigenvalues_fromEigen;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>::EigenvectorsType eigenvectors_fromEigen;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covM);
    if (eigensolver.info() != Eigen::Success) {
      eigenvalues_fromEigen = eigenvalues_fromEigen.Zero();
      eigenvectors_fromEigen = eigenvectors_fromEigen.Zero();
    } else {
      eigenvalues_fromEigen = eigensolver.eigenvalues();
      eigenvectors_fromEigen = eigensolver.eigenvectors();
    }

    // Compute the spread in the both spaces.
    for (size_t i = 0; i < N; ++i) {
      fillPoint(layerClusters[trackster.vertices(i)]);
      sigmas += weight * (point - barycenter).cwiseAbs2();
      Eigen::Vector3d point_transformed = eigenvectors_fromEigen * (point - barycenter);
      if (energyWeight && trackster.raw_energy())
        weight =
            (layerClusters[trackster.vertices(i)].energy() / trackster.vertex_multiplicity(i)) / trackster.raw_energy();
      sigmasEigen += weight * (point_transformed.cwiseAbs2());
    }
    sigmas /= (1. - weights2_sum);
    sigmasEigen /= (1. - weights2_sum);

    // Add trackster attributes
    trackster.setBarycenter(ticl::Trackster::Vector(barycenter));
    trackster.setTimeAndError(timeTrackster.first, timeTrackster.second);
    trackster.fillPCAVariables(
        eigenvalues_fromEigen, eigenvectors_fromEigen, sigmas, sigmasEigen, 3, ticl::Trackster::PCAOrdering::ascending);

    LogDebug("TrackstersPCA") << "Use energy weighting: " << energyWeight << std::endl;
    LogDebug("TrackstersPCA") << "\nTrackster characteristics: " << std::endl;
    LogDebug("TrackstersPCA") << "Size: " << N << std::endl;
    LogDebug("TrackstersPCA") << "Energy: " << trackster.raw_energy() << std::endl;
    LogDebug("TrackstersPCA") << "raw_pt: " << trackster.raw_pt() << std::endl;
    LogDebug("TrackstersPCA") << "Means:          " << barycenter[0] << ", " << barycenter[1] << ", " << barycenter[2]
                              << std::endl;
    LogDebug("TrackstersPCA") << "Time:          " << trackster.time() << " +/- " << trackster.timeError() << std::endl;
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
