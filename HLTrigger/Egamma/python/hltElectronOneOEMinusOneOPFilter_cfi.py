import FWCore.ParameterSet.Config as cms

hltElectronOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsL1NonIsoForHLT"),
    electronIsolatedProducer = cms.InputTag("pixelMatchElectronsL1IsoForHLT"),
    barrelcut = cms.double(0.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltElectronPixelMatchFilter"),
    endcapcut = cms.double(0.03),
    saveTags = cms.bool( False )
)


