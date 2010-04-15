import FWCore.ParameterSet.Config as cms

# Make sure to enable the "TimerService" service at your top-level cfg!
# by including something like the following:
# service = TimerService {
#  untracked bool useCPUtime = true // set to false for wall-clock-time
# }
# This is the module that stores in the Event the timing info
myTimer = cms.EDProducer("Timer",
    # whether to include timing info about Timer itself
    includeSelf = cms.untracked.bool(False)
)


