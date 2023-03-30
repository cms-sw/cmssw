#ifndef HeterogeneousCore_AlpakaInterface_interface_CopyToHost_h
#define HeterogeneousCore_AlpakaInterface_interface_CopyToHost_h

// TODO: this utility class is specific to CMSSW, but needs to be in a
// package that is suitable as DataFormat dependence

namespace cms::alpakatools {
  /**
   * This class template needs to be specialized for each device-side
   * Event data product so that the framework can implicitly copy the
   * device-side data product to the host memory. The specialization
   * is expected to define static copyAsync() function as in the
   * following example
   *
   * \code
   * template <>
   * struct CopyToHost<ExampleDeviceProduct> {
   *   template <typename TQueue>
   *   static ExampleHostProduct copyAsync(TQueue& queue, ExampleDeviceProduct const& deviceData) {
   *     // construct ExampleHostProduct
   *     // asynchronous copy deviceData to the ExampleHostProduct object
   *     // return ExampleHostProduct object by value
   *   }
   * };
   * \endcode
   *
   * The copyAsync() function should not explicitly synchronize the
   * queue. The ExampleDeviceProduct and ExampleHostProduct can be the
   * same type, if they internally are able to handle the memory
   * allocation difference between host and device.
   */
  template <typename TDeviceData>
  struct CopyToHost;
}  // namespace cms::alpakatools

#endif
