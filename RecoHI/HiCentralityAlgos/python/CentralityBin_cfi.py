import FWCore.ParameterSet.Config as cms

centralityBin = cms.EDProducer('CentralityBinProducer',
    Centrality = cms.InputTag("hiCentrality"),
    centralityVariable = cms.string("HFtowers"),
    nonDefaultGlauberModel = cms.string(""),
    pPbRunFlip = cms.uint32(99999999),
)

# foo bar baz
# KQLI7nkzMomDV
# sbry30ZmiUH8g
