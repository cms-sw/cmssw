import FWCore.ParameterSet.Config as cms

# creation of shallow clone candidates
caloJetCollectionClone = cms.EDProducer("CaloJetShallowCloneProducer",
    src = cms.InputTag("allLayer0Jets")
)

# association of the flavour with the shallow clones
jetPartonAssociation = cms.EDFilter("CandJetFlavourIdentifier",
    jets = cms.InputTag("caloJetCollectionClone"),
    debug = cms.bool(False),
    coneSizeToAssociate = cms.double(0.3),
    vetoFlavour = cms.vstring(),
    physicsDefinition = cms.bool(False)
)

# default PAT sequence for jet flavour identification
patJetFlavourId = cms.Sequence(caloJetCollectionClone*jetPartonAssociation)

