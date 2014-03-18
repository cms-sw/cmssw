import FWCore.ParameterSet.Config as cms

pileupJPTJetIdProducer = cms.EDProducer("PileupJPTJetIdProducer",
   jets = cms.InputTag("ak4JPTJetsL1L2L3"),
   Verbosity = cms.int32(0),
   tmvaWeightsCentral = cms.string('RecoJets/JetProducers/data/TMVAClassification_PUJetID_JPT_BDTG.weights.xml.gz'),
   tmvaWeightsForward = cms.string('RecoJets/JetProducers/data/TMVAClassification_PUJetID_JPT_BDTG.weights_F.xml.gz'),
   tmvaMethod = cms.string('BDTG method')
)

