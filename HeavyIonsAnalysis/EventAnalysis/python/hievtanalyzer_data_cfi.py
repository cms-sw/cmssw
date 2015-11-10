import FWCore.ParameterSet.Config as cms

hiEvtAnalyzer = cms.EDAnalyzer('HiEvtAnalyzer',
   CentralitySrc    = cms.InputTag("hiCentrality"),
   CentralityBinSrc = cms.InputTag("centralityBin","HFtowers"),
   EvtPlane         = cms.InputTag("hiEvtPlane"),
   EvtPlaneFlat     = cms.InputTag("hiEvtPlaneFlat",""),   
   HiMC             = cms.InputTag("heavyIon"),                            
   Vertex           = cms.InputTag("hiSelectedVertex"),
   PixelClusSrc     = cms.InputTag("siPixelClusters"),
   doCentrality     = cms.bool(True),
   doEvtPlane       = cms.bool(True),
   doEvtPlaneFlat   = cms.bool(False),                               
   doVertex         = cms.bool(True),
   doPixel          = cms.bool(False),
   doMC             = cms.bool(False),
   evtPlaneLevel    = cms.int32(0)
)
