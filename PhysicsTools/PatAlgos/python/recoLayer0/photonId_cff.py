import FWCore.ParameterSet.Config as cms

patAODPhotonID = cms.EDFilter("PhotonIDConverter",
    src = cms.InputTag("photons"),
    photonID = cms.InputTag("PhotonIDProd","PhotonAssociatedID")
)

layer0PhotonID = cms.EDFilter("CandValueMapSkimmerPhotonID",
    collection  = cms.InputTag("allLayer0Photons"),
    backrefs    = cms.InputTag("allLayer0Photons"),
    association = cms.InputTag("patAODPhotonID"),
) 

## define the sequence, so we have consistent naming conventions
patLayer0PhotonID = cms.Sequence( layer0PhotonID )

