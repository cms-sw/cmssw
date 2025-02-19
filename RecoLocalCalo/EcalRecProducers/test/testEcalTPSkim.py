import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.load("RecoLocalCalo.EcalRecProducers.ecalTPSkim_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:in.root')
                            )

process.p = cms.Path( process.ecalTPSkim )

