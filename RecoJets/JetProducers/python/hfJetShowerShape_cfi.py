import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.hfJetShowerShapeDefault_cfi import hfJetShowerShapeDefault

hfJetShowerShape = hfJetShowerShapeDefault.clone(
#    jets = cms.InputTag(''),
#    vertices = cms.InputTag(''),
    jetPtThreshold = cms.double(25),
    jetEtaThreshold = cms.double(2.9),
    hfTowerEtaWidth = cms.double(0.175),
    hfTowerPhiWidth = cms.double(0.175),
    vertexRecoEffcy = cms.double(0.7),
    offsetPerPU = cms.double(0.4),
    jetReferenceRadius = cms.double(0.4),
    stripPtThreshold = cms.double(10),
    widthPtThreshold = cms.double(3)
)


