import FWCore.ParameterSet.Config as cms

hltFourVectorClient = cms.EDFilter("FourVectorHLTClient",
    prescaleLS = cms.untracked.int32(-1),
    monitorDir = cms.untracked.string(''),
    prescaleEvt = cms.untracked.int32(1)
)


