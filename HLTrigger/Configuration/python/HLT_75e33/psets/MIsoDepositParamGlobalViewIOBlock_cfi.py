import FWCore.ParameterSet.Config as cms

MIsoDepositParamGlobalViewIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MultipleDepositsFlag = cms.bool(False),
    MuonTrackRefType = cms.string('bestTrkSta'),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons")
)