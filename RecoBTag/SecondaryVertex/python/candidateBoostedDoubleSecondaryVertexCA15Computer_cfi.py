import FWCore.ParameterSet.Config as cms

candidateBoostedDoubleSecondaryVertexCA15Computer = cms.ESProducer("CandidateBoostedDoubleSecondaryVertexESProducer",
    beta = cms.double(1.0),
    R0 = cms.double(1.5),
    maxSVDeltaRToJet = cms.double(1.3),
    weightFile = cms.FileInPath('RecoBTag/SecondaryVertex/data/BoostedDoubleSV_CA15_BDT.weights.xml.gz')
)
