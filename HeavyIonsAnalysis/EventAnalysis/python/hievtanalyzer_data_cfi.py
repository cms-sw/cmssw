import FWCore.ParameterSet.Config as cms

hiEvtAnalyzer = cms.EDAnalyzer('HiEvtAnalyzer',
   Centrality    = cms.InputTag("hiCentrality"),
   CentralityBin = cms.InputTag("centralityBin"),
   EvtPlane      = cms.InputTag("hiEvtPlane","recoLevel"),
   EvtPlaneFlat  = cms.InputTag("hiEvtPlaneFlat",""),                               
   Vertex        = cms.InputTag("hiSelectedVertex"),
   HiMC          = cms.InputTag("heavyIon"),
   doCentrality  = cms.bool(True),
   doEvtPlane    = cms.bool(True),
   doEvtPlaneFlat= cms.bool(False),                               
   doVertex      = cms.bool(True),
   doMC          = cms.bool(False)
)
