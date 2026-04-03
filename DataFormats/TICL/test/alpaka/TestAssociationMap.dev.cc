
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICL/interface/AssociationMap.h"
#include "DataFormats/TICL/interface/FillAssociator.h"

#include <alpaka/alpaka.hpp>
#include <ranges>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {
  template <typename TKey, typename TMapped>
  struct TestKernel {
    ALPAKA_FN_ACC void operator()(const Acc1D& acc, ticl::AssociationMapView<TKey, TMapped> map, bool* result) const {
      for (auto idx : alpaka::uniformElements(acc, map.size() / 2)) {
        *result &= (map[0][idx] % 2 != 0);
        *result &= (map[1][idx] % 2 == 0);
      }
      *result &= (map.count(0) == 50);
      *result &= (map.count(1) == 50);
      *result &= map.contains(0);
      *result &= map.contains(1);
      *result &= (map.keys() == 2);
    }
  };
}  // namespace

TEST_CASE("Construct and fill an association map") {
  const auto& devices = cms::alpakatools::devices<Platform>();

  SECTION("Test default map") {
    for (const auto& device : devices) {
      Queue queue(device);

      const auto nkeys = 2u;
      const auto nvalues = 100u;
      PortableCollection<Device, ticl::AssociationMap<>> map(queue, nvalues, nkeys);
      auto host_values = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
      auto host_associations = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
      std::ranges::copy(std::views::iota(0u, nvalues), host_values.data());
      for (auto i : std::views::iota(0u, nvalues))
        host_associations[i] = (i % 2 == 0);
      auto device_values = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nvalues);
      auto device_associations = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nvalues);
      alpaka::memcpy(queue, device_values, host_values);
      alpaka::memcpy(queue, device_associations, host_associations);
      ticl::associator::fill<Acc1D>(queue,
                                    map.view(),
                                    std::span<const uint32_t>{device_associations.data(), nvalues},
                                    std::span<const uint32_t>{device_values.data(), nvalues});
      auto offsets = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nkeys);
      auto values = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
      alpaka::memcpy(
          queue, offsets, cms::alpakatools::make_device_view(queue, map.view().offsets().keys_offsets().data(), nkeys));
      alpaka::memcpy(
          queue, values, cms::alpakatools::make_device_view(queue, map.view().content().values().data(), nvalues));

      auto device_result = cms::alpakatools::make_device_buffer<bool>(queue);
      alpaka::memset(queue, device_result, 1);
      const auto blocksize = 1024u;
      const auto gridsize = cms::alpakatools::divide_up_by(nvalues, blocksize);
      auto work_division = cms::alpakatools::make_workdiv<Acc1D>(gridsize, blocksize);
      alpaka::exec<Acc1D>(queue, work_division, TestKernel<uint32_t, uint32_t>{}, map.view(), device_result.data());
      auto host_result = cms::alpakatools::make_host_buffer<bool>(queue);
      alpaka::memcpy(queue, host_result, device_result);
      alpaka::wait(queue);

      CHECK(*host_result);

      // check content back on CPU
      CHECK(offsets[0] == 50);
      CHECK(offsets[1] == 100);
      for (auto i = 0u; i < nvalues; ++i) {
        if (i < nvalues / 2) {
          CHECK(values[i] % 2 != 0);
        } else {
          CHECK(values[i] % 2 == 0);
        }
      }
    }
  }
  SECTION("Test map with specialized templates") {
    for (const auto& device : devices) {
      Queue queue(device);

      const auto nkeys = 2;
      const auto nvalues = 100;
      PortableCollection<Device, ticl::AssociationMap<int, int>> map(queue, nvalues, nkeys);
      auto host_values = cms::alpakatools::make_host_buffer<int[]>(queue, nvalues);
      auto host_associations = cms::alpakatools::make_host_buffer<int[]>(queue, nvalues);
      std::ranges::copy(std::views::iota(0, nvalues), host_values.data());
      for (auto i : std::views::iota(0, nvalues))
        host_associations[i] = (i % 2 == 0);
      auto device_values = cms::alpakatools::make_device_buffer<int[]>(queue, nvalues);
      auto device_associations = cms::alpakatools::make_device_buffer<int[]>(queue, nvalues);
      alpaka::memcpy(queue, device_values, host_values);
      alpaka::memcpy(queue, device_associations, host_associations);
      ticl::associator::fill<Acc1D>(queue,
                                    map.view(),
                                    std::span<const int>{device_associations.data(), static_cast<std::size_t>(nvalues)},
                                    std::span<const int>{device_values.data(), static_cast<std::size_t>(nvalues)});
      auto offsets = cms::alpakatools::make_host_buffer<int[]>(queue, nkeys);
      auto values = cms::alpakatools::make_host_buffer<int[]>(queue, nvalues);
      alpaka::memcpy(
          queue, offsets, cms::alpakatools::make_device_view(queue, map.view().offsets().keys_offsets().data(), nkeys));
      alpaka::memcpy(
          queue, values, cms::alpakatools::make_device_view(queue, map.view().content().values().data(), nvalues));

      auto device_result = cms::alpakatools::make_device_buffer<bool>(queue);
      alpaka::memset(queue, device_result, 1);
      const auto blocksize = 1024u;
      const auto gridsize = cms::alpakatools::divide_up_by(nvalues, blocksize);
      auto work_division = cms::alpakatools::make_workdiv<Acc1D>(gridsize, blocksize);
      alpaka::exec<Acc1D>(queue, work_division, TestKernel<int, int>{}, map.view(), device_result.data());
      auto host_result = cms::alpakatools::make_host_buffer<bool>(queue);
      alpaka::memcpy(queue, host_result, device_result);
      alpaka::wait(queue);

      CHECK(*host_result);

      // check content back on CPU
      CHECK(offsets[0] == 50);
      CHECK(offsets[1] == 100);
      for (auto i = 0u; i < nvalues; ++i) {
        if (i < nvalues / 2) {
          CHECK(values[i] % 2 != 0);
        } else {
          CHECK(values[i] % 2 == 0);
        }
      }
    }
  }
}
