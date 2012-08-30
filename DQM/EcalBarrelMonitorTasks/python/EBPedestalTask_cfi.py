import FWCore.ParameterSet.Config as cms

ecalBarrelPedestalTask = cms.EDAnalyzer("EBPedestalTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EBDigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    MGPAGains = cms.untracked.vint32(1, 6, 12),
    MGPAGainsPN = cms.untracked.vint32(1, 16)
)

