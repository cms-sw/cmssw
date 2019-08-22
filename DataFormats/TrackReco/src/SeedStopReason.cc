#include "DataFormats/TrackReco/interface/SeedStopReason.h"

static_assert(sizeof(SeedStopReasonName::SeedStopReasonName) / sizeof(std::string) ==
                  static_cast<unsigned int>(SeedStopReason::SIZE),
              "SeedStopReason enum and SeedStopReasonName are out of synch");
