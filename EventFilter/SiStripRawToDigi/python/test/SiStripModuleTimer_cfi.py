import FWCore.ParameterSet.Config as cms

ModuleTimer = cms.EDFilter("SiStripModuleTimer",
    ModuleLabels = cms.untracked.vstring()
)


