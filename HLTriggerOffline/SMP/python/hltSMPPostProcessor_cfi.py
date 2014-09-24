import FWCore.ParameterSet.Config as cms

hltSMPPostProcessor  = cms.EDAnalyzer("DQMGenericClient",
    subDirs           = cms.untracked.vstring('HLT/SMP/*'),
    verbose           = cms.untracked.uint32(2),
    outputFileName    = cms.untracked.string(''),
    resolution        = cms.vstring(),                                    
    efficiency        = cms.vstring(),
    efficiencyProfile = cms.untracked.vstring(),
)

