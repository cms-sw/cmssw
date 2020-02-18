#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackstersPCA.h"
#include "TPrincipal.h"

#include <iostream>

#include <Eigen/Dense>

#define LogDebug(X) std::cout

void ticl::assignPCAtoTracksters(std::vector<Trackster> & tracksters,
    const std::vector<reco::CaloCluster> &layerClusters, double z_limit_em, bool energyWeight) {
  energyWeight = false;
  TPrincipal pca(3, "N");
  LogDebug("TrackstersPCA") << "-------" << std::endl;
  for (auto &trackster : tracksters) {
    double point[3] = {0};
    auto fillPoint = [&](const reco::CaloCluster & c, const float weight=1) {point[0] = weight*c.x(); point[1] = weight*c.y(); point[2] = weight*c.z();};
    pca.Clear();

    trackster.raw_energy = 0.;
    trackster.raw_em_energy = 0.;
    trackster.raw_pt = 0.;
    trackster.raw_em_pt = 0.;
    size_t N = trackster.vertices.size();
    for (size_t i = 0; i < N; ++i) {
      auto fraction = 1.f / trackster.vertex_multiplicity[i];
      trackster.raw_energy += layerClusters[trackster.vertices[i]].energy() * fraction;
      if (std::abs(layerClusters[trackster.vertices[i]].z()) <= z_limit_em)
        trackster.raw_em_energy += layerClusters[trackster.vertices[i]].energy() * fraction;
    }

    //float weight = 1;
    float weight = 1.f / N;
    float weights_sum = 0.f;
    double barycenter[3] = {0};
    for (size_t i = 0; i < N; ++i) {
      fillPoint(layerClusters[trackster.vertices[i]], 1); // ROOT's PCA can not deal with weight
      pca.AddRow(point);

      if (energyWeight && trackster.raw_energy)
        weight = (layerClusters[trackster.vertices[i]].energy() / trackster.vertex_multiplicity[i]) / trackster.raw_energy;
      weights_sum += weight;
      fillPoint(layerClusters[trackster.vertices[i]], weight);

      for (size_t j=0; j<3; ++j)
        barycenter[j] += point[j];
    }
    for (size_t j=0; j<3; ++j)
      barycenter[j] /= weights_sum;

    pca.MakePrincipals();
    const auto & barycenter_ = *(pca.GetMeanValues());
    float weights2_sum = 0.f;
    double sigmas[3] = {0};
    double sigmasPCA[3] = {0};
    for (size_t i = 0; i < N; ++i) {
      fillPoint(layerClusters[trackster.vertices[i]]);
      double p[3];
      pca.X2P(point, p);
      if (energyWeight && trackster.raw_energy)
        weight = (layerClusters[trackster.vertices[i]].energy() / trackster.vertex_multiplicity[i]) / trackster.raw_energy;
      weights2_sum += weight*weight;
      for (size_t j=0; j<3; ++j) {
        sigmas[j] += weight * (point[j] - barycenter[j]) * (point[j] - barycenter[j]);
        sigmasPCA[j] += (p[j] - barycenter[j]) * (p[j] - barycenter[j]);
      }
    }

    // Add trackster attributes
    trackster.barycenter = ticl::Trackster::Vector(barycenter[0],
        barycenter[1],
        barycenter[2]);
    double sigmas_[3] = {0};
    for (size_t i=0; i<3; ++i) {
      sigmas[i] = std::sqrt(sigmas[i]/(weights_sum - weights2_sum/weights_sum));
      sigmas_[i] = (float)(*(pca.GetSigmas()))[i];
      sigmasPCA[i] = std::sqrt(sigmasPCA[i]/N);
      trackster.sigmas[i] = sigmas[i];
      trackster.sigmasPCA[i] = sigmasPCA[i];
      trackster.eigenvalues[i] = (float)(*(pca.GetEigenValues()))[i];
      trackster.eigenvectors[i] = ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][i],
        (*(pca.GetEigenVectors()))[1][i],
        (*(pca.GetEigenVectors()))[2][i] );
    }
    if (trackster.eigenvectors[0].z() * trackster.barycenter.z() < 0.0) {
      trackster.eigenvectors[0] = -ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][0],
          (*(pca.GetEigenVectors()))[1][0],
          (*(pca.GetEigenVectors()))[2][0] );
    }
    auto norm = std::sqrt(trackster.eigenvectors[0].Unit().perp2());
    trackster.raw_pt = norm * trackster.raw_energy;
    trackster.raw_em_pt = norm * trackster.raw_em_energy;
    const auto & eigenvectors = *(pca.GetEigenVectors());
    const auto & eigenvalues = *(pca.GetEigenValues());

    // Eigen way
    Eigen::Matrix3d covM = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < N; ++i) {
      fillPoint(layerClusters[trackster.vertices[i]]);
      if (energyWeight && trackster.raw_energy)
        weight = (layerClusters[trackster.vertices[i]].energy() / trackster.vertex_multiplicity[i]) / trackster.raw_energy;
      for (size_t x=0; x<3; ++x)
        for (size_t y=0; y<=x; ++y) {
          covM(x,y) += weight*(point[x] - barycenter[x])*(point[y] - barycenter[y]);
          if (x != y)
            covM(y,x) = covM(x,y);
        }
    }
    covM *= 1. / (weights_sum - weights2_sum/weights_sum);

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


    LogDebug("TrackstersPCA") << "\nTrackster characteristics: " << std::endl;
    LogDebug("TrackstersPCA") << "Size: " << N << std::endl;
    LogDebug("TrackstersPCA") << "Energy: " << trackster.raw_energy << std::endl;
    LogDebug("TrackstersPCA") << "Means: " << barycenter[0] << ", " << barycenter[1] << ", " << barycenter[2] << std::endl;
    LogDebug("TrackstersPCA") << "Means from PCA: " << barycenter_[0] << ", " << barycenter_[1] << ", " << barycenter_[2] << std::endl;
    LogDebug("TrackstersPCA") << "Weights sum: " << weights_sum << std::endl;
    LogDebug("TrackstersPCA") << "EigenValues: " << eigenvalues[0] << ", " << eigenvalues[1] << ", " << eigenvalues[2]  << std::endl;
    LogDebug("TrackstersPCA") << "EigenValues from Eigen: " << eigenvalues_fromEigen[0] << ", " << eigenvalues_fromEigen[1] << ", " << eigenvalues_fromEigen[2] << std::endl;
    LogDebug("TrackstersPCA") << "EigenVector 1: " << eigenvectors(0, 0) << ", " << eigenvectors(1, 0) << ", " << eigenvectors(2, 0) <<std::endl;
    LogDebug("TrackstersPCA") << "EigenVector 1 from Eigen: " << eigenvectors_fromEigen(0, 0) << ", " << eigenvectors_fromEigen(1, 0) << ", " << eigenvectors_fromEigen(2, 0) <<std::endl;
    LogDebug("TrackstersPCA") << "EigenVector 2: " << eigenvectors(0, 1) << ", " << eigenvectors(1, 1) << ", " << eigenvectors(2, 1) <<std::endl;
    LogDebug("TrackstersPCA") << "EigenVector 2 from Eigen: " << eigenvectors_fromEigen(0, 1) << ", " << eigenvectors_fromEigen(1, 1) << ", " << eigenvectors_fromEigen(2, 1) <<std::endl;
    LogDebug("TrackstersPCA") << "EigenVector 3: " << eigenvectors(0, 2) << ", " << eigenvectors(1, 2) << ", " << eigenvectors(2, 2) <<std::endl;
    LogDebug("TrackstersPCA") << "EigenVector 3 from Eigen: " << eigenvectors_fromEigen(0, 2) << ", " << eigenvectors_fromEigen(1, 2) << ", " << eigenvectors_fromEigen(2, 2) <<std::endl;
    LogDebug("TrackstersPCA") << "Original sigmas: " << sigmas[0] << ", " << sigmas[1] << ", " << sigmas[2] << std::endl;
    LogDebug("TrackstersPCA") << "Original sigmas from PCA: " << sigmas_[0] << ", " << sigmas_[1] << ", " << sigmas_[2] << std::endl;
    LogDebug("TrackstersPCA") << "covM: \n" << covM << std::endl;
    LogDebug("TrackstersPCA") << "SigmasPCA: " << sigmasPCA[0] << ", " << sigmasPCA[1] << ", " << sigmasPCA[2] << std::endl;
  }
}
