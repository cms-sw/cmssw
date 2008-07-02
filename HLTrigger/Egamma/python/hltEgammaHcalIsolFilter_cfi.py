import FWCore.ParameterSet.Config as cms

hltEgammaHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    doIsolated = cms.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltSingleEgammaHcalIsol"),
    candTag = cms.InputTag("hltSingleEgammaEtFilter")
)


