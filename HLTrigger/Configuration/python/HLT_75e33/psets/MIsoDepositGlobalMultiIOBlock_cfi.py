import FWCore.ParameterSet.Config as cms

MIsoDepositGlobalMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    InputType = cms.string('TrackCollection'),
    MultipleDepositsFlag = cms.bool(True),
    MuonTrackRefType = cms.string('track'),
    inputMuonCollection = cms.InputTag("globalMuons")
)