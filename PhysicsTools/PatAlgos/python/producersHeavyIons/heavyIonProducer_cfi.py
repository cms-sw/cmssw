import FWCore.ParameterSet.Config as cms

heavyIon = cms.EDProducer("PATHeavyIonProducer",
  doReco     = cms.bool(True),
  doMC       = cms.bool(True),
  centrality = cms.InputTag("hiCentrality","recoBased"),
  evtPlane   = cms.InputTag("hiEvtPlane","recoLevel"),
  generators = cms.vstring("generator")
)



