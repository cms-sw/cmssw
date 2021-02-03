import FWCore.ParameterSet.Config as cms

MIsoDepositGlobalIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    InputType = cms.string('TrackCollection'),
    MultipleDepositsFlag = cms.bool(False),
    MuonTrackRefType = cms.string('track'),
    inputMuonCollection = cms.InputTag("globalMuons")
)