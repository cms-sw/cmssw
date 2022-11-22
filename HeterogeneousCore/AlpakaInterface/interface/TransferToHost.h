#ifndef HeterogeneousCore_AlpakaInterface_interface_TransferToHost_h
#define HeterogeneousCore_AlpakaInterface_interface_TransferToHost_h

// TODO: better package?

namespace cms::alpakatools {
  // TODO: would a more informative error message from compiler than "indeterminate type" be helpful?
  template <typename TDeviceData>
  struct TransferToHost;

  // specialization expected to define
  // using HostDataType = <corresponding host data type>
  //
  // template <typename TQueue>
  // static HostDataType transferAsync(TQueue& queue, TDeviceData const& deviceData);
  //
  // The function should allocate a HostDataType object and launch the
  // transfers
}  // namespace cms::alpakatools

#endif
