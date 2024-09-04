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
   *
   * Data products that contain pointers to memory elsewhere in the
   * data product need those pointers to be updated after the copy
   * from device-to-host completes. While such data structures are
   * generally discouraged, such an update of the data product can be
   * implemented (without any additional synchronization) with an
   * optional postCopy() static member function in the CopyToHost
   * specialization. The postCopy() is called for the host-side data
   * product after the copy operations enqueued in the copyAsync()
   * have finished. Following the example above, the expected
   * signature is
   * \code
   * template <>
   * struct CopyToHost<ExampleDeviceProduct> {
   *   // copyAsync() definition from above
   *
   *   static void postCopy(ExampleHostProduct& obj) {
   *     // modify obj
   *     // any modifications must be such that the postCopy() can be
   *     // skipped when the obj originates from the host (i.e. on CPU backends)
   *   }
   * };
   * \endcode
   */
  template <typename TDeviceData>
  struct CopyToHost;
}  // namespace cms::alpakatools

#endif
