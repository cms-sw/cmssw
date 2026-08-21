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
      ticl::associator::fill<Acc1D>(
          queue, map.view(), std::span<const uint32_t>{device_associations}, std::span<const uint32_t>{device_values});
      auto offsets = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, map.view().offsets().keys_offsets().size());
      auto values = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
      alpaka::memcpy(queue, offsets, cms::alpakatools::make_device_view(queue, map.view().offsets().keys_offsets()));
      alpaka::memcpy(queue, values, cms::alpakatools::make_device_view(queue, map.view().content().values()));

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
      CHECK(offsets[0] == 0);
      CHECK(offsets[1] == 50);
      CHECK(offsets[2] == 100);
      for (auto i = 0u; i < nvalues; ++i) {
        if (i < nvalues / 2) {
          CHECK(values[i] % 2 != 0);
        } else {
          CHECK(values[i] % 2 == 0);
        }
      }
      SECTION("Test partially filled map") {
        PortableCollection<Device, ticl::AssociationMap<>> partial_map(queue, nvalues, nkeys);
        auto partial_map_view = partial_map.view();
        ticl::associator::fill<Acc1D>(queue,
                                      partial_map_view,
                                      std::span<const uint32_t>{device_associations.data(), nvalues / 2},
                                      std::span<const uint32_t>{device_values.data(), nvalues / 2});
        auto partial_offsets =
            cms::alpakatools::make_host_buffer<uint32_t[]>(queue, partial_map.view().offsets().keys_offsets().size());
        auto partial_values = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues / 2);
        alpaka::memcpy(queue,
                       partial_offsets,
                       cms::alpakatools::make_device_view(queue, partial_map.view().offsets().keys_offsets()));
        alpaka::memcpy(queue,
                       partial_values,
                       cms::alpakatools::make_device_view(queue, partial_map.view().content().values(), nvalues / 2));
        alpaka::wait(queue);

        CHECK(partial_offsets[0] == 0);
        CHECK(partial_offsets[1] == nvalues / 4);
        CHECK(partial_offsets[2] == nvalues / 2);
        for (auto i = 0u; i < nvalues / 2; ++i) {
          if (i < nvalues / 4) {
            CHECK(partial_values[i] % 2 != 0);
          } else {
            CHECK(partial_values[i] % 2 == 0);
          }
        }
      }
      SECTION("Test deep copy") {
        PortableCollection<Device, ticl::AssociationMap<>> new_map(queue, nvalues, nkeys);
        new_map.deepCopy(queue, map.const_view());

        alpaka::memset(queue, offsets, 0x0);
        alpaka::memset(queue, values, 0x0);
        alpaka::memcpy(
            queue, offsets, cms::alpakatools::make_device_view(queue, new_map.view().offsets().keys_offsets()));
        alpaka::memcpy(queue, values, cms::alpakatools::make_device_view(queue, new_map.view().content().values()));
        alpaka::wait(queue);

        CHECK(offsets[0] == 0);
        CHECK(offsets[1] == 50);
        CHECK(offsets[2] == 100);
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
      auto offsets = cms::alpakatools::make_host_buffer<int[]>(queue, map.view().offsets().keys_offsets().size());
      auto values = cms::alpakatools::make_host_buffer<int[]>(queue, nvalues);
      alpaka::memcpy(queue, offsets, cms::alpakatools::make_device_view(queue, map.view().offsets().keys_offsets()));
      alpaka::memcpy(queue, values, cms::alpakatools::make_device_view(queue, map.view().content().values()));

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
      CHECK(offsets[0] == 0);
      CHECK(offsets[1] == 50);
      CHECK(offsets[2] == 100);
      for (auto i = 0u; i < nvalues; ++i) {
        if (i < nvalues / 2) {
          CHECK(values[i] % 2 != 0);
        } else {
          CHECK(values[i] % 2 == 0);
        }
      }
      SECTION("Test deep copy") {
        PortableCollection<Device, ticl::AssociationMap<int, int>> new_map(queue, nvalues, nkeys);
        new_map.deepCopy(queue, map.const_view());

        alpaka::memset(queue, offsets, 0x0);
        alpaka::memset(queue, values, 0x0);
        alpaka::memcpy(
            queue, offsets, cms::alpakatools::make_device_view(queue, new_map.view().offsets().keys_offsets()));
        alpaka::memcpy(queue, values, cms::alpakatools::make_device_view(queue, new_map.view().content().values()));
        alpaka::wait(queue);

        CHECK(offsets[0] == 0);
        CHECK(offsets[1] == 50);
        CHECK(offsets[2] == 100);
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
}

TEST_CASE("Test different numbers of keys") {
  const auto& devices = cms::alpakatools::devices<Platform>();

  SECTION("Test map with keys close to alignments") {
    for (const auto& device : devices) {
      Queue queue(device);

      const auto nvalues = 100u;
      auto host_values = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
      std::ranges::copy(std::views::iota(0u, nvalues), host_values.data());

      const auto nkeys1 = 30;
      const auto nkeys2 = 31;
      const auto nkeys3 = 32;
      auto host_associations1 = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
      auto host_associations2 = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
      auto host_associations3 = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
      for (auto i : std::views::iota(0u, nvalues)) {
        host_associations1[i] = (i < static_cast<unsigned>(nkeys1)) ? i : nkeys1 - 1;
        host_associations2[i] = (i < static_cast<unsigned>(nkeys2)) ? i : nkeys2 - 1;
        host_associations3[i] = (i < static_cast<unsigned>(nkeys3)) ? i : nkeys3 - 1;
      }
      auto device_values = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nvalues);
      auto device_associations1 = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nvalues);
      auto device_associations2 = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nvalues);
      auto device_associations3 = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nvalues);
      alpaka::memcpy(queue, device_values, host_values);
      alpaka::memcpy(queue, device_associations1, host_associations1);
      alpaka::memcpy(queue, device_associations2, host_associations2);
      alpaka::memcpy(queue, device_associations3, host_associations3);

      PortableCollection<Device, ticl::AssociationMap<>> map1(queue, nvalues, nkeys1);
      PortableCollection<Device, ticl::AssociationMap<>> map2(queue, nvalues, nkeys2);
      PortableCollection<Device, ticl::AssociationMap<>> map3(queue, nvalues, nkeys3);
      ticl::associator::fill<Acc1D>(queue,
                                    map1.view(),
                                    std::span<const uint32_t>{device_associations1},
                                    std::span<const uint32_t>{device_values});
      ticl::associator::fill<Acc1D>(queue,
                                    map2.view(),
                                    std::span<const uint32_t>{device_associations2},
                                    std::span<const uint32_t>{device_values});
      ticl::associator::fill<Acc1D>(queue,
                                    map3.view(),
                                    std::span<const uint32_t>{device_associations3},
                                    std::span<const uint32_t>{device_values});

      // every key below `nkeys - 1` gets exactly the single value equal to its own index;
      // all the overflow values (i >= nkeys - 1) are collected, in order, under the last key
      auto checkMap = [&](auto& map, int nkeys) {
        auto offsets =
            cms::alpakatools::make_host_buffer<uint32_t[]>(queue, map.view().offsets().keys_offsets().size());
        auto values = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nvalues);
        alpaka::memcpy(queue, offsets, cms::alpakatools::make_device_view(queue, map.view().offsets().keys_offsets()));
        alpaka::memcpy(queue, values, cms::alpakatools::make_device_view(queue, map.view().content().values()));
        alpaka::wait(queue);

        const auto last_key = static_cast<uint32_t>(nkeys - 1);
        CHECK(offsets[0] == 0);
        for (auto k = 0u; k < last_key; ++k) {
          CHECK(offsets[k + 1] - offsets[k] == 1);
          CHECK(values[offsets[k]] == k);
        }
        CHECK(offsets[nkeys] == nvalues);
        for (auto i = last_key; i < nvalues; ++i) {
          CHECK(values[offsets[last_key] + (i - last_key)] >= last_key);
        }
      };
      checkMap(map1, nkeys1);
      checkMap(map2, nkeys2);
      checkMap(map3, nkeys3);
    }
  }
}
