import FWCore.ParameterSet.Config as cms

# SiStripOfflineDQM
modSiStripOfflineDQM = cms.EDFilter("SiStripOfflineDQM",
    nQTestEventsDelay = cms.untracked.int32(10),
    bVerbose = cms.untracked.bool(False)
)


