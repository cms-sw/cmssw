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
    iDesc.add<double>("rho_c", 6.);
    iDesc.add<std::vector<double>>("dc", {2., 2., 2.});
    iDesc.add<std::vector<double>>("dm", {1.8, 1.8, 2.});
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
