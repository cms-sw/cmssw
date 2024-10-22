#ifndef HeterogeneousCore_Common_interface_PlatformStatus_h
#define HeterogeneousCore_Common_interface_PlatformStatus_h

// Please note: these values must be kept in sync with HeterogeneousCore/Common/python/PlatformStatus.py

enum PlatformStatus : int {
  Success = 0,
  PlatformNotAvailable = 1,  // the platform is not available for this architecture, OS or compiler
  RuntimeNotAvailable = 2,   // the runtime could not be initialised
  DevicesNotAvailable = 3,   // there are no visible, usable devices
};

#endif  // HeterogeneousCore_Common_interface_PlatformStatus_h
