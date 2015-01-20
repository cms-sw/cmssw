import FWCore.ParameterSet.Config as cms

centralityBin = cms.EDProducer('CentralityBinProducer',
    Centrality = cms.InputTag("hiCentrality"),
    centralityVariable = cms.string("HFtowers"),
    nonDefaultGlauberModel = cms.string(""),
    pPbRunFlip = cms.uint(99999999),
)

