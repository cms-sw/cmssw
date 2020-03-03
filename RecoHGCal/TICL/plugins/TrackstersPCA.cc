#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackstersPCA.h"
#include "TPrincipal.h"

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

void ticl::assignPCAtoTracksters(std::vector<Trackster> &tracksters,
                                 const std::vector<reco::CaloCluster> &layerClusters,
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
    trackster.raw_energy = 0.;
    trackster.raw_em_energy = 0.;
    trackster.raw_pt = 0.;
    trackster.raw_em_pt = 0.;

    size_t N = trackster.vertices.size();
    float weight = 1.f / N;
    float weights2_sum = 0.f;
    Eigen::Vector3d sigmas;
    sigmas << 0., 0., 0.;
    Eigen::Vector3d sigmasEigen;
    sigmasEigen << 0., 0., 0.;
    Eigen::Matrix3d covM = Eigen::Matrix3d::Zero();

    for (size_t i = 0; i < N; ++i) {
      auto fraction = 1.f / trackster.vertex_multiplicity[i];
      trackster.raw_energy += layerClusters[trackster.vertices[i]].energy() * fraction;
      if (std::abs(layerClusters[trackster.vertices[i]].z()) <= z_limit_em)
        trackster.raw_em_energy += layerClusters[trackster.vertices[i]].energy() * fraction;

      // Compute the weighted barycenter.
      if (energyWeight)
        weight = layerClusters[trackster.vertices[i]].energy() * fraction;
      fillPoint(layerClusters[trackster.vertices[i]], weight);
      for (size_t j = 0; j < 3; ++j)
        barycenter[j] += point[j];
    }
    if (energyWeight && trackster.raw_energy)
      barycenter /= trackster.raw_energy;

    // Compute the Covariance Matrix and the sum of the squared weights, used
    // to compute the correct normalization.
    // The barycenter has to be known.
    for (size_t i = 0; i < N; ++i) {
      fillPoint(layerClusters[trackster.vertices[i]]);
      if (energyWeight && trackster.raw_energy)
        weight =
            (layerClusters[trackster.vertices[i]].energy() / trackster.vertex_multiplicity[i]) / trackster.raw_energy;
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
      fillPoint(layerClusters[trackster.vertices[i]]);
      sigmas += weight * (point - barycenter).cwiseAbs2();
      Eigen::Vector3d point_transformed = eigenvectors_fromEigen * (point - barycenter);
      if (energyWeight && trackster.raw_energy)
        weight =
            (layerClusters[trackster.vertices[i]].energy() / trackster.vertex_multiplicity[i]) / trackster.raw_energy;
      sigmasEigen += weight * (point_transformed.cwiseAbs2());
    }
    sigmas /= (1. - weights2_sum);
    sigmasEigen /= (1. - weights2_sum);

    // Add trackster attributes
    trackster.barycenter = ticl::Trackster::Vector(barycenter[0], barycenter[1], barycenter[2]);
    for (size_t i = 0; i < 3; ++i) {
      sigmas[i] = std::sqrt(sigmas[i]);
      sigmasEigen[i] = std::sqrt(sigmasEigen[i]);
      trackster.sigmas[i] = sigmas[i];
      trackster.sigmasPCA[i] = sigmasEigen[i];
      // Reverse the order, since Eigen gives back the eigevalues in increasing order.
      trackster.eigenvalues[i] = (float)eigenvalues_fromEigen[2 - i];
      trackster.eigenvectors[i] = ticl::Trackster::Vector(
          eigenvectors_fromEigen(0, 2 - i), eigenvectors_fromEigen(1, 2 - i), eigenvectors_fromEigen(2, 2 - i));
    }
    if (trackster.eigenvectors[0].z() * trackster.barycenter.z() < 0.0) {
      trackster.eigenvectors[0] = -ticl::Trackster::Vector(
          eigenvectors_fromEigen(0, 2), eigenvectors_fromEigen(1, 2), eigenvectors_fromEigen(2, 2));
    }
    trackster.raw_pt = std::sqrt((trackster.eigenvectors[0].Unit() * trackster.raw_energy).perp2());
    trackster.raw_em_pt = std::sqrt((trackster.eigenvectors[0].Unit() * trackster.raw_em_energy).perp2());

    LogDebug("TrackstersPCA") << "Use energy weighting: " << energyWeight << std::endl;
    LogDebug("TrackstersPCA") << "\nTrackster characteristics: " << std::endl;
    LogDebug("TrackstersPCA") << "Size: " << N << std::endl;
    LogDebug("TrackstersPCA") << "Energy: " << trackster.raw_energy << std::endl;
    LogDebug("TrackstersPCA") << "raw_pt: " << trackster.raw_pt << std::endl;
    LogDebug("TrackstersPCA") << "Means:          " << barycenter[0] << ", " << barycenter[1] << ", " << barycenter[2]
                              << std::endl;
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
