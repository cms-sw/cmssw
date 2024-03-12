import FWCore.ParameterSet.Config as cms

matchGenHFHadron = cms.EDProducer("GenHFHadronMatcher",
    genParticles = cms.InputTag('genParticles'),
    jetFlavourInfos = cms.InputTag("genJetFlavourInfos"),
    flavour = cms.int32(5),
    onlyJetClusteredHadrons = cms.bool(True),
    noBBbarResonances = cms.bool(False),
)



# foo bar baz
# TKsTIWtncoqTM
# NE74nlB7Hmien
