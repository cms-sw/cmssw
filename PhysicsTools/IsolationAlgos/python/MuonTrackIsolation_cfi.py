import FWCore.ParameterSet.Config as cms

#
# MuonPtIsolationProducer
# -> isolation of muons against the sum of pt of tracks 
#
MuonPtIsolation = cms.EDProducer("MuonPtIsolationProducer",
    src = cms.InputTag("globalMuons"),
    d0Max = cms.double(20.0),
    dRMin = cms.double(0.0),
    dRMax = cms.double(0.3),
    elements = cms.InputTag("generalTracks"),
    ptMin = cms.double(0.0),
    dzMax = cms.double(20.0)
)

MuonPtIsolation05 = cms.EDProducer("MuonPtIsolationProducer",
    dzMax = cms.double(20.0),
    src = cms.InputTag("globalMuons"),
    dRMin = cms.double(0.0),
    elements = cms.InputTag("generalTracks"),
    dRMax = cms.double(0.5)
)


