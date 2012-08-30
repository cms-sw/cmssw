import FWCore.ParameterSet.Config as cms

ecalEndcapPedestalTask = cms.EDAnalyzer("EEPedestalTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    mergeRuns = cms.untracked.bool(False),    
    enableCleanup = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EEDigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    MGPAGains = cms.untracked.vint32(1, 6, 12),
    MGPAGainsPN = cms.untracked.vint32(1, 16)                                        
)

