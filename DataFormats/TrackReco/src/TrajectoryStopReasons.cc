#include "DataFormats/TrackReco/interface/TrajectoryStopReasons.h"

static_assert(sizeof(StopReasonName::StopReasonName)/sizeof(std::string) == static_cast<unsigned int>(StopReason::SIZE), "StopReason enum and StopReasonName are out of synch");
