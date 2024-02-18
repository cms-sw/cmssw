#ifndef HeterogeneousCore_AlpakaInterface_interface_CopyToDevice_h
#define HeterogeneousCore_AlpakaInterface_interface_CopyToDevice_h

// TODO: this utility class is specific to CMSSW, but needs to be in a
// package that is suitable as DataFormat dependence

namespace cms::alpakatools {
  /**
   * This class template needs to be specialized for each host-side
   * EventSetup data product that should be implicitly copied to the
   * device memory. The specialization is expected to define static
   * copyAsync() function as in the following example
   *
   * \code
   * template <>
   * struct CopyToDevice<ExampleHostProduct> {
   *   template <typename TQueue>
   *   static auto copyAsync(TQueue& queue, ExampleHostProduct const& hostData) {
   *     // construct ExampleDeviceProduct corresponding the device of the TQueue
   *     // asynchronous copy hostData to the ExampleDeviceProduct object
   *     // return ExampleDeviceProduct object by value
   *   }
   * };
   * \endcode
   *
   * The copyAsync() function should not explicitly synchronize the
   * queue. The ExampleHostProduct and ExampleDevicxeProduct can be the
   * same type, if they internally are able to handle the memory
   * allocation difference between host and device.
   */
  template <typename THostData>
  struct CopyToDevice;
}  // namespace cms::alpakatools

// specialize to Alpaka buffer
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
namespace cms::alpakatools {
  // Note: can't do partial specializations along
  // - CopyToDevice<host_buffer<TObject>>
  // - CopyToDevice<alpaka::Buf<alpaka_common::DevHost, TObject, alpaka_common::Dim0D, alpaka_common::Idx>e
  // because both host_buffer and alpaka::Buf use trait-style
  // indirection that prevents template argument type deduction
  template <typename TObject>
  struct CopyToDevice<alpaka::BufCpu<TObject, alpaka_common::Dim0D, alpaka_common::Idx>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, host_buffer<TObject> const& src) {
      using TDevice = alpaka::Dev<TQueue>;
      auto dst = make_device_buffer<TObject>(queue);
      alpaka::memcpy(queue, dst, src);
      return dst;
    }
  };

  template <typename TObject>
  struct CopyToDevice<alpaka::BufCpu<TObject, alpaka_common::Dim1D, alpaka_common::Idx>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, host_buffer<TObject[]> const& src) {
      using TDevice = alpaka::Dev<TQueue>;
      auto dst = make_device_buffer<TObject[]>(queue, alpaka::getExtentProduct(src));
      alpaka::memcpy(queue, dst, src);
      return dst;
    }
  };
}  // namespace cms::alpakatools

#endif
