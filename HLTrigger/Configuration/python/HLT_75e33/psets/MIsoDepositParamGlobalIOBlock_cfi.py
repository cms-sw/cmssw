import FWCore.ParameterSet.Config as cms

MIsoDepositParamGlobalIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MultipleDepositsFlag = cms.bool(False),
    MuonTrackRefType = cms.string('track'),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons")
)