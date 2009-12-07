import FWCore.ParameterSet.Config as cms

muonSkim = cms.EDFilter( "MuonFilter",
                         muonsLabel      = cms.InputTag("muons"),
                         caloMuonsLabel  = cms.InputTag("calomuons"),

                         selectL1Trigger      = cms.bool(False),
                         selectOnDTHits       = cms.bool(True),
                         selectOnRPCHits      = cms.bool(True),
                         selectMuons          = cms.bool(True),
                         selectCaloMuons      = cms.bool(False)
)

