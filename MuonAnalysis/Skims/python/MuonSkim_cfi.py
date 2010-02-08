import FWCore.ParameterSet.Config as cms

muonSkim = cms.EDFilter( "MuonFilter",
                         muonsLabel      = cms.InputTag("muons"),
                         caloMuonsLabel  = cms.InputTag("calomuons"),

                         selectOnDTHits       = cms.bool(True),
                         selectMuons          = cms.bool(True),
                         selectCaloMuons      = cms.bool(False)
)

