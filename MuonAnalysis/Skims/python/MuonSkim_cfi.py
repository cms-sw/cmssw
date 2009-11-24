import FWCore.ParameterSet.Config as cms

muonSkim = cms.EDFilter( "MuonFilter",
    muonTag        = cms.InputTag("muons"),
    caloMuonTag    = cms.InputTag("calomuons"),
    acceptMuon     = cms.untracked.bool(True),
    acceptCalo     = cms.untracked.bool(False)
)

