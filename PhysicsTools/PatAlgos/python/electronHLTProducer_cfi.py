import FWCore.ParameterSet.Config as cms

HLT1Electron = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter","","HLT"),
    triggerName = cms.string('HLT1Electron')
)

HLT1ElectronRelaxed = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter","","HLT"),
    triggerName = cms.string('HLT1ElectronRelaxed')
)

HLT2Electron = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter","","HLT"),
    triggerName = cms.string('HLT2Electron')
)

HLT2ElectronRelaxed = cms.EDProducer("PATHLTProducer",
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    filterName = cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter","","HLT"),
    triggerName = cms.string('HLT2ElectronRelaxed')
)

electronHLTProducer = cms.Sequence(HLT1Electron*HLT1ElectronRelaxed*HLT2Electron*HLT2ElectronRelaxed)

