import FWCore.ParameterSet.Config as cms

hltCaloMET = cms.EDProducer("CaloMETProducer",
    alias = cms.string('RawCaloMET'),
    calculateSignificance = cms.bool(False),
    globalThreshold = cms.double(0.3),
    noHF = cms.bool(False),
    src = cms.InputTag("towerMaker")
)
