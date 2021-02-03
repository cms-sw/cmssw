import FWCore.ParameterSet.Config as cms

MIsoDepositParamGlobalViewMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MultipleDepositsFlag = cms.bool(True),
    MuonTrackRefType = cms.string('bestTrkSta'),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons")
)