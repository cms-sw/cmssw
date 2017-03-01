import FWCore.ParameterSet.Config as cms

l1ElectronRecoTree = cms.EDAnalyzer("L1ElectronRecoTreeProducer",
   maxElectron                          = cms.uint32(20),
   ElectronTag                          = cms.untracked.InputTag("gedGsfElectrons"),
   RhoTag                               = cms.untracked.InputTag("fixedGridRhoFastjetAllCalo"),
   VerticesTag                          = cms.untracked.InputTag("offlinePrimaryVertices"),#vertices"),
   ConversionsTag                       = cms.untracked.InputTag("conversions"),
   BeamSpotTag                          = cms.untracked.InputTag("offlineBeamSpot")


)

