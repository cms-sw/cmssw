import FWCore.ParameterSet.Config as cms

pileupJPTJetIdProducer = cms.EDProducer("PileupJPTJetIdProducer",
   jets = cms.InputTag("ak5JPTJetsL1L2L3"),
   Verbosity = cms.int32(0),
   tmvaWeightsCentral = cms.string('RecoJets/JetProducers/data/TMVAClassification_BDTG.weights.xml'),
   tmvaWeightsForward = cms.string('RecoJets/JetProducers/data/TMVAClassification_BDTG.weights_F.xml'),
   tmvaMethod = cms.string('BDTG method')
)

