import FWCore.ParameterSet.Config as cms

patAODJetTracksAssociator = cms.EDFilter("JetTracksAssociationValueMap",
    src    = cms.InputTag("iterativeCone5CaloJets"),         ## the AOD jets given as input to the PAT jet cleaner
    tracks = cms.InputTag("ic5JetTracksAssociatorAtVertex"), ## any JetTracksAssociation
    cut = cms.string('') # e.g. normalizedChi2 < 5
)

## Re-key from AOD jets to PAT Layer 0 jets
layer0JetTracksAssociator = cms.EDFilter("CandValueMapSkimmerTrackRefs",
    collection  = cms.InputTag("allLayer0Jets"),
    backrefs    = cms.InputTag("allLayer0Jets"),
    association = cms.InputTag("patAODJetTracksAssociator"),
)

## Compute JET Charge
layer0JetCharge = cms.EDFilter("JetChargeValueMap",
    src                  = cms.InputTag("allLayer0Jets"),             ## The Jets
    jetTracksAssociation = cms.InputTag("layer0JetTracksAssociator"), ## NOTE: must be something from JetTracksAssociationValueMap
    # -- JetCharge parameters --
    var = cms.string('Pt'),
    exp = cms.double(1.0)
)

patLayer0JetTracksCharge = cms.Sequence(patAODJetTracksAssociator * layer0JetTracksAssociator * layer0JetCharge)

