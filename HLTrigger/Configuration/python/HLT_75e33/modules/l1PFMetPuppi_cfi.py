import FWCore.ParameterSet.Config as cms

l1PFMetPuppi = cms.EDProducer("PFMETProducer",
    alias = cms.string('@module_label'),
    applyWeight = cms.bool(False),
    calculateSignificance = cms.bool(False),
    globalThreshold = cms.double(0),
    mightGet = cms.optional.untracked.vstring,
    parameters = cms.PSet(

    ),
    src = cms.InputTag("l1pfCandidates","Puppi"),
    srcJetResPhi = cms.optional.string,
    srcJetResPt = cms.optional.string,
    srcJetSF = cms.optional.string,
    srcJets = cms.optional.InputTag,
    srcLeptons = cms.optional.VInputTag,
    srcRho = cms.optional.InputTag,
    srcWeights = cms.InputTag("")
)
