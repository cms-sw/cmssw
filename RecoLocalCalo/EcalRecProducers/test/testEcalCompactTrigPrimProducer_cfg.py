import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.ecalCompactTrigPrimProducerTest = cms.EDAnalyzer("EcalCompactTrigPrimProducerTest",
                                                     #tpDigiColl = cms.InputTag("simEcalTriggerPrimitiveDigis"),
                                                     tpDigiColl = cms.InputTag("ecalDigis:EcalTriggerPrimitives"),
                                                     tpRecColl  = cms.InputTag("ecalCompactTrigPrim")
                                                     )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:in.root')
                            )

#if TP rec hits not in input data sample:
#process.load("RecoLocalCalo.EcalRecProducers.ecalCompactTrigPrim_cfi")
# #process.ecalCompactTrigPrim.inColl = cms.InputTag("simEcalTriggerPrimitiveDigis")
#process.p = cms.Path(process.ecalCompactTrigPrim*process.ecalCompactTrigPrimProducerTest)
#else
process.p = cms.Path(process.ecalCompactTrigPrimProducerTest)

