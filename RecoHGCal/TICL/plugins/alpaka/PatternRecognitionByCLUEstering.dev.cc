#include "DataFormats/HGCalReco/interface/HGCalSoAClusters.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoHGCal/TICL/interface/alpaka/PatternRecognitionAlgoBase.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "CLUEstering/CLUEstering.hpp"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoHGCal/TICL/plugins/alpaka/PatternRecognitionByCLUEstering.h"

#include <iterator>
#include <unordered_map>
#include <vector>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace ticl = ::ticl;

  void PatternRecognitionByCLUEstering::makeTracksters(Queue& queue,
                                                       const HGCalSoAClustersDeviceCollection& lc,
                                                       std::vector<ticl::Trackster>& tracksters) {
    auto* x = const_cast<float*>(lc.view().x().data());
    auto* y = const_cast<float*>(lc.view().y().data());
    auto* z = const_cast<float*>(lc.view().z().data());
    auto* E = const_cast<float*>(lc.view().energy().data());
    // std::unordered_map<float, std::vector<int>> map;

    // for (int i = 0; i < lc->metadata().size(); ++i) {
    //   map[z[i]].push_back(i);
    // }

    const int32_t n = static_cast<int32_t>(lc->metadata().size());
    if (n > 0) {
      auto d_clIndex = cms::alpakatools::make_device_buffer<int[]>(queue, n);  // temporary buffer needed by CLUEstering
      auto dp_clIndex = const_cast<int*>(d_clIndex.data());
      clue::PointsDevice<3> d_points(queue, n, x, y, z, E, dp_clIndex);
      // if (m_verbosity) {
      //   for (int iLC = 0; iLC < n; ++iLC) {
      //     std::cout << "( " << x[iLC] << ", " << y[iLC] << ", " << z[iLC] << ", " << E[iLC] << " )" << std::endl;
      //   }
      // }

      clue::Clusterer<3> clusterer(queue, m_dc, m_rhoc, m_dm);
      clusterer.make_clusters(queue, d_points);
      // create hosts points and do the copy
      clue::PointsHost<3> h_points(queue, n);
      clue::copyToHost(queue, h_points, d_points);
      alpaka::wait(queue);

      // get LCs indices in tracksters and fill the trackster collection
      const auto tsMap = clue::get_clusters(h_points);
      tracksters.resize(tsMap.size());
      for (auto i = 0ul; i < tsMap.size(); ++i) {
        const auto [beginLC, endLC] = tsMap.equal_range(i);
        std::copy(beginLC, endLC, std::back_inserter(tracksters[i].vertices()));
        tracksters[i].vertex_multiplicity().resize(tracksters[i].vertices().size(), 1);
      }

      alpaka::memcpy(queue,
                     cms::alpakatools::make_host_view(h_points.view().coords[0], n),
                     cms::alpakatools::make_device_view(alpaka::getDev(queue), x, n),
                     static_cast<uint32_t>(n));
      alpaka::memcpy(queue,
                     cms::alpakatools::make_host_view(h_points.view().coords[1], n),
                     cms::alpakatools::make_device_view(alpaka::getDev(queue), y, n),
                     static_cast<uint32_t>(n));
      alpaka::memcpy(queue,
                     cms::alpakatools::make_host_view(h_points.view().coords[2], n),
                     cms::alpakatools::make_device_view(alpaka::getDev(queue), z, n),
                     static_cast<uint32_t>(n));
      alpaka::memcpy(queue,
                     cms::alpakatools::make_host_view(h_points.weights(), n),
                     cms::alpakatools::make_device_view(alpaka::getDev(queue), E, n),
                     static_cast<uint32_t>(n));
      alpaka::wait(queue);

      // compute trackster properties
      // TODO: merge with previous loop
      bool energyWeight = true;
      auto xHost = h_points.coords(0).data();
      auto yHost = h_points.coords(1).data();
      auto zHost = h_points.coords(2).data();
      auto EHost = h_points.weights();
      // if (m_verbosity) {
      //   std::cout << "Event Number of LCs " << n << std::endl;
      //   for (const auto& [Z, indices] : map) {
      //     std::cout << "z = " << Z << " -> Clusters : ";
      //     for (auto i : indices)
      //       std::cout << "\t( " << xHost[i] << ", " << yHost[i] << ", " << zHost[i] << ", " << EHost[i] << ")"
      //                 << std::endl;
      //     std::cout << std::endl;
      //   }
      // }
      for (auto& trackster : tracksters) {
        size_t N = trackster.vertices().size();
        if (N == 0)
          continue;

        Eigen::Vector3f point;
        point << 0., 0., 0.;
        Eigen::Vector3f barycenter;
        barycenter << 0., 0., 0.;

        auto fillPoint = [&](const float x, const float y, const float z, const float weight = 1.f) {
          point[0] = weight * x;
          point[1] = weight * y;
          point[2] = weight * z;
        };

        // Initialize this trackster with default, dummy values
        trackster.setRawEnergy(0.f);
        trackster.setRawEmEnergy(0.f);
        trackster.setRawPt(0.f);
        trackster.setRawEmPt(0.f);

        float weight = 1.f / N;

        std::vector<float> layerClusterEnergies;

        for (size_t i = 0; i < N; ++i) {
          auto lcIdx = trackster.vertices(i);
          auto fraction = 1.f / trackster.vertex_multiplicity(i);
          trackster.addToRawEnergy(EHost[lcIdx] * fraction);
          // trackster.addToRawEmEnergy(EHost[lcIdx] * fraction);

          // Compute the weighted barycenter.
          if (energyWeight)
            weight = EHost[lcIdx] * fraction;
          fillPoint(xHost[lcIdx], yHost[lcIdx], zHost[lcIdx], weight);
          for (size_t j = 0; j < 3; ++j)
            barycenter[j] += point[j];

          layerClusterEnergies.push_back(EHost[lcIdx]);
        }
        float raw_energy = trackster.raw_energy();
        float inv_raw_energy = 1.f / raw_energy;
        if (energyWeight)
          barycenter *= inv_raw_energy;
        trackster.setBarycenter(ticl::Trackster::Vector(barycenter));

        trackster.calculateRawPt();
        trackster.calculateRawEmPt();
        // if (m_verbosity) {
        //   std::cout << "  LC in TS: ";
        //   for (const auto& lc : trackster.vertices())
        //     std::cout << lc << " ";
        //   std::cout << std::endl;
        //   std::cout << "  energy raw: " << trackster.raw_energy() << std::endl;
        //   std::cout << "  barycenter: " << trackster.barycenter().x() << ", " << trackster.barycenter().y() << ", "
        //             << trackster.barycenter().z() << std::endl;
        // }
      }
    }
  }

  void PatternRecognitionByCLUEstering::fillPSetDescription(::edm::ParameterSetDescription& iDesc) {
    // iDesc.add<int>("algo_verbosity", 0);
    // iDesc.add<std::vector<double>>("criticalDensity", {4, 4, 4})->setComment("in GeV");
    // iDesc.add<std::vector<double>>("criticalSelfDensity", {0.15, 0.15, 0.15} /* roughly 1/(densitySiblingLayers+1) */)
    //     ->setComment("Minimum ratio of self_energy/local_density to become a seed.");
    // iDesc.add<std::vector<int>>("densitySiblingLayers", {3, 3, 3})
    //     ->setComment(
    //         "inclusive, layers to consider while computing local density and searching for nearestHigher higher");
    // iDesc.add<std::vector<double>>("densityEtaPhiDistanceSqr", {0.0008, 0.0008, 0.0008})
    //     ->setComment("in eta,phi space, distance to consider for local density");
    // iDesc.add<std::vector<double>>("densityXYDistanceSqr", {3.24, 3.24, 3.24})
    //     ->setComment("in cm, distance on the transverse plane to consider for local density");
    // iDesc.add<std::vector<double>>("kernelDensityFactor", {0.2, 0.2, 0.2})
    //     ->setComment("Kernel factor to be applied to other LC while computing the local density");
    // iDesc.add<bool>("densityOnSameLayer", false);
    // iDesc.add<bool>("nearestHigherOnSameLayer", false)
    //     ->setComment("Allow the nearestHigher to be located on the same layer");
    // iDesc.add<bool>("useAbsoluteProjectiveScale", true)
    //     ->setComment("Express all cuts in terms of r/z*z_0{,phi} projective variables");
    // iDesc.add<bool>("useClusterDimensionXY", false)
    //     ->setComment(
    //         "Boolean. If true use the estimated cluster radius to determine the cluster compatibility while computing "
    //         "the local density");
    // iDesc.add<bool>("rescaleDensityByZ", false)
    //     ->setComment(
    //         "Rescale local density by the extension of the Z 'volume' explored. The transvere dimension is, at "
    //         "present, "
    //         "fixed and factored out.");
    // iDesc.add<std::vector<double>>("criticalEtaPhiDistance", {0.025, 0.025, 0.025})
    //     ->setComment("Minimal distance in eta,phi space from nearestHigher to become a seed");
    // iDesc.add<std::vector<double>>("criticalXYDistance", {1.8, 1.8, 1.8})
    //     ->setComment("Minimal distance in cm on the XY plane from nearestHigher to become a seed");
    // iDesc.add<std::vector<int>>("criticalZDistanceLyr", {5, 5, 5})
    //     ->setComment("Minimal distance in layers along the Z axis from nearestHigher to become a seed");
    // iDesc.add<std::vector<double>>("outlierMultiplier", {2, 2, 2})
    //     ->setComment("Minimal distance in transverse space from nearestHigher to become an outlier");
    // iDesc.add<std::vector<int>>("minNumLayerCluster", {2, 2, 2})->setComment("Not Inclusive");
    // iDesc.add<bool>("doPidCut", false);
    // iDesc.add<double>("cutHadProb", 0.5);
    // iDesc.add<bool>("computeLocalTime", false);
    // iDesc.add<bool>("usePCACleaning", false)->setComment("Enable PCA cleaning alorithm");
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
