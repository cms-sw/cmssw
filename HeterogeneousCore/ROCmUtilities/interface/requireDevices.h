#ifndef HeterogeneousCore_ROCmUtilities_interface_requireDevices_h
#define HeterogeneousCore_ROCmUtilities_interface_requireDevices_h

/**
 * These functions are meant to be called only from unit tests.
 */
namespace cms {
  namespace rocmtest {

    /// In presence of ROCm devices, return true; otherwise print message and return false
    bool testDevices();

    /// Print message and exit if there are no ROCm devices
    void requireDevices();

  }  // namespace rocmtest
}  // namespace cms

#endif  // HeterogeneousCore_ROCmUtilities_interface_requireDevices_h
