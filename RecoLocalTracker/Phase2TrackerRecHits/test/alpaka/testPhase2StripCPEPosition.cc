// The Phase2StripCPE cluster position computed on the device (phase2StripCPE::localPosition) against
// RectangularPixelPhase2Topology::localX / localY on the host, as Phase2StripCPE uses them: bitwise on the CPU
// backends, within kGpuTolerance on the GPU backends (the device compiler may contract a*b + c*d differently).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEModuleParams.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

#include "testPhase2StripCPEPosition.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using testPhase2StripCPEPosition::ClusterInput;

namespace {

  constexpr float kGpuTolerance = 1e-5f;  // cm

  struct TopologyConfig {
    int nRows, nColumns;
    float pitchX, pitchY;
    int rowsPerRoc, columnsPerRoc, bigPixelsPerRocX, bigPixelsPerRocY;
    float bigPitchX, bigPitchY;
    int rocsX, rocsY;
  };

}  // namespace

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::fprintf(stderr,
                 "No devices available for the " EDM_STRINGIZE(
                     ALPAKA_ACCELERATOR_NAMESPACE) " backend, the test will be skipped.\n");
    exit(EXIT_FAILURE);
  }
  constexpr bool isCpu = std::is_same_v<Device, alpaka::DevCpu>;

  // OT-like topologies (2S strips, PS strips, PS macro-pixels, and macro-pixels with big pixels of a different pitch),
  // so that every branch of localX / localY is used
  const std::vector<TopologyConfig> topologies = {{1016, 2, 0.009f, 5.025f, 127, 1, 0, 0, 0.009f, 5.025f, 8, 2},
                                                  {960, 2, 0.01f, 2.4f, 120, 1, 0, 0, 0.01f, 2.4f, 8, 2},
                                                  {960, 32, 0.01f, 0.1467f, 120, 16, 0, 0, 0.01f, 0.1467f, 8, 2},
                                                  {960, 32, 0.01f, 0.1467f, 120, 16, 1, 1, 0.0213f, 0.25f, 8, 2}};
  const std::vector<float> coveredStrips = {-0.37f, 0.f, 0.21f, 1.3f};
  const std::vector<uint16_t> sizes = {1, 2, 3, 4, 7};

  int failures = 0;
  for (auto const& device : devices) {
    Queue queue(device);
    float maxDifference = 0.f;
    long long compared = 0, different = 0;
    for (auto const& config : topologies) {
      const RectangularPixelPhase2Topology topology(config.nRows,
                                                    config.nColumns,
                                                    config.pitchX,
                                                    config.pitchY,
                                                    config.rowsPerRoc,
                                                    config.columnsPerRoc,
                                                    config.bigPixelsPerRocX,
                                                    config.bigPixelsPerRocY,
                                                    config.bigPitchX,
                                                    config.bigPitchY,
                                                    config.rocsX,
                                                    config.rocsY);
      std::vector<ClusterInput> inputs;
      for (int strip = 0; strip < config.nRows; ++strip)
        for (int column = 0; column < config.nColumns; ++column)
          for (auto size : sizes)
            inputs.push_back({uint32_t(strip), uint32_t(column), size});
      const uint32_t nInputs = inputs.size();
      auto clustersHost = cms::alpakatools::make_host_buffer<ClusterInput[]>(queue, nInputs);
      std::memcpy(clustersHost.data(), inputs.data(), nInputs * sizeof(ClusterInput));
      auto clustersDevice = cms::alpakatools::make_device_buffer<ClusterInput[]>(queue, nInputs);
      auto positionsDevice = cms::alpakatools::make_device_buffer<phase2StripCPE::LocalPosition[]>(queue, nInputs);
      auto positionsHost = cms::alpakatools::make_host_buffer<phase2StripCPE::LocalPosition[]>(queue, nInputs);
      alpaka::memcpy(queue, clustersDevice, clustersHost);
      for (float covered : coveredStrips) {
        const phase2StripCPE::ModuleParams params = phase2StripCPE::moduleParams(topology, covered);
        testPhase2StripCPEPosition::localPositions(
            queue, params, clustersDevice.data(), positionsDevice.data(), nInputs);
        alpaka::memcpy(queue, positionsHost, positionsDevice);
        alpaka::wait(queue);
        for (uint32_t index = 0; index < nInputs; ++index) {
          auto const& input = inputs[index];
          const Phase2TrackerCluster1D cluster(input.firstStrip, input.column, input.size);
          // Phase2StripCPE::localParameters
          const float referenceX = topology.localX(cluster.center() - 0.5f * covered);
          const float referenceY = topology.localY(float(cluster.column()) + 0.5f);
          for (auto [reference, value] :
               {std::pair{referenceX, positionsHost[index].x}, std::pair{referenceY, positionsHost[index].y}}) {
            ++compared;
            if (std::memcmp(&reference, &value, sizeof(float)) != 0) {
              ++different;
              maxDifference = std::max(maxDifference, std::abs(reference - value));
            }
          }
        }
      }
    }
    const bool pass = isCpu ? different == 0 : maxDifference <= kGpuTolerance;
    std::printf("%s %s: %lld coordinates, %lld not bitwise identical to the host topology, max |difference| %g cm\n",
                pass ? "PASS" : "FAIL",
                alpaka::getName(device).c_str(),
                compared,
                different,
                maxDifference);
    if (not pass)
      ++failures;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
