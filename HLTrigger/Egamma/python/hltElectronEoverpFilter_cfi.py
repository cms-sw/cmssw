import FWCore.ParameterSet.Config as cms

hltElectronEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    electronIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    eoverpendcapcut = cms.double(2.45),
    ncandcut = cms.int32(1),
    eoverpbarrelcut = cms.double(1.5),
    candTag = cms.InputTag("hltElectronPixelMatchFilter"),
    saveTags = cms.bool( False )
)


