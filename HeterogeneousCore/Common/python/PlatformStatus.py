import enum

# Please note: these values must be kept in sync with HeterogeneousCore/Common/interface/PlatformStatus.h

class PlatformStatus(enum.IntEnum):
    Success = 0 
    PlatformNotAvailable = 1    # the platform is not available for this architecture, OS or compiler
    RuntimeNotAvailable = 2     # the runtime could not be initialised
    DevicesNotAvailable = 3     # there are no visible, usable devices
