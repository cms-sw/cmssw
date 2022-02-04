import FWCore.ParameterSet.Config as cms

hltPixelClustersMultiplicity = cms.EDProducer("HLTSiPixelClusterMultiplicityValueProducer",
    defaultValue = cms.double(-1.0),
    mightGet = cms.optional.untracked.vstring,
    src = cms.InputTag("siPixelClusters")
)
