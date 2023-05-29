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

#endif
