import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
MIsoDepositViewMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("muons"),
    MultipleDepositsFlag = cms.bool(True),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('bestGlbTrkSta')
)
MIsoDepositViewIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("muons"),
    MultipleDepositsFlag = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('bestGlbTrkSta')
)
MIsoDepositParamGlobalViewMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
    MultipleDepositsFlag = cms.bool(True),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('bestGlbTrkSta')
)
MIsoDepositParamGlobalViewIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
    MultipleDepositsFlag = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('bestGlbTrkSta')
)
MIsoDepositParamGlobalMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
    MultipleDepositsFlag = cms.bool(True),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('track')
)
MIsoDepositParamGlobalIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
    MultipleDepositsFlag = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('track')
)
MIsoDepositGlobalIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("globalMuons"),
    MultipleDepositsFlag = cms.bool(False),
    InputType = cms.string('TrackCollection'),
    MuonTrackRefType = cms.string('track')
)
MIsoDepositGlobalMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("globalMuons"),
    MultipleDepositsFlag = cms.bool(True),
    InputType = cms.string('TrackCollection'),
    MuonTrackRefType = cms.string('track')
)


