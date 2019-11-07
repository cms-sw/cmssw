#include "TrackstersPCA.h"
#include "TPrincipal.h"

#include <iostream>

void ticl::assignPCAtoTracksters(std::vector<Trackster> & tracksters,
    const std::vector<reco::CaloCluster> &layerClusters) {
  TPrincipal pca(3, "");
  std::cout << "-------" << std::endl;
  for (auto &trackster : tracksters) {
    pca.Clear();
    for (size_t i = 0; i < trackster.vertices.size(); ++i) {
      auto fraction = 1.f / trackster.vertex_multiplicity[i];
      trackster.raw_energy += layerClusters[trackster.vertices[i]].energy() * fraction;
      double point[3] = {
        layerClusters[trackster.vertices[i]].x(),
        layerClusters[trackster.vertices[i]].y(),
        layerClusters[trackster.vertices[i]].z()
      };
      pca.AddRow(&point[0]);
    }
    pca.MakePrincipals();
    trackster.barycenter = ticl::Trackster::Vector((*(pca.GetMeanValues()))[0],
        (*(pca.GetMeanValues()))[1],
        (*(pca.GetMeanValues()))[2]);
    trackster.eigenvalues[0] = (float)(*(pca.GetEigenValues()))[0];
    trackster.eigenvalues[1] = (float)(*(pca.GetEigenValues()))[1];
    trackster.eigenvalues[2] = (float)(*(pca.GetEigenValues()))[2];
    trackster.eigenvectors[0] = ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][0],
        (*(pca.GetEigenVectors()))[1][0],
        (*(pca.GetEigenVectors()))[2][0] );
    trackster.eigenvectors[1] = ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][1],
        (*(pca.GetEigenVectors()))[1][1],
        (*(pca.GetEigenVectors()))[2][1] );
    trackster.eigenvectors[2] = ticl::Trackster::Vector((*(pca.GetEigenVectors()))[0][2],
        (*(pca.GetEigenVectors()))[1][2],
        (*(pca.GetEigenVectors()))[2][2] );
    trackster.sigmas[0] = (float)(*(pca.GetSigmas()))[0];
    trackster.sigmas[1] = (float)(*(pca.GetSigmas()))[1];
    trackster.sigmas[2] = (float)(*(pca.GetSigmas()))[2];
    const auto & mean = *(pca.GetMeanValues());
    const auto & eigenvectors = *(pca.GetEigenVectors());
    const auto & eigenvalues = *(pca.GetEigenValues());
    const auto & sigmas = *(pca.GetSigmas());
    std::cout << "Trackster characteristics: " << std::endl;
    std::cout << "Size: " << trackster.vertices.size() << std::endl;
    std::cout << "Mean: " << mean[0] << ", " << mean[1] << ", " << mean[2] << std::endl;
    std::cout << "EigenValues: " << eigenvalues[0] << ", " << eigenvalues[1] << ", " << eigenvalues[2]  << std::endl;
    std::cout << "EigeVectors 1: " << eigenvectors(0, 0) << ", " << eigenvectors(1, 0) << ", " << eigenvectors(2, 0) <<std::endl;
    std::cout << "EigeVectors 2: " << eigenvectors(0, 1) << ", " << eigenvectors(1, 1) << ", " << eigenvectors(2, 1) <<std::endl;
    std::cout << "EigeVectors 3: " << eigenvectors(0, 2) << ", " << eigenvectors(1, 2) << ", " << eigenvectors(2, 2) <<std::endl;
    std::cout << "Sigmas: " << sigmas[0] << ", " << sigmas[1] << ", " << sigmas[2] << std::endl;
  }
}
