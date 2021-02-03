import FWCore.ParameterSet.Config as cms

MIsoDepositParamGlobalMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MultipleDepositsFlag = cms.bool(True),
    MuonTrackRefType = cms.string('track'),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons")
)