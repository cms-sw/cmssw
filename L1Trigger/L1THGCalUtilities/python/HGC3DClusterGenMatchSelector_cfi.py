import FWCore.ParameterSet.Config as cms

hgc3DClusterGenMatchSelector = cms.EDProducer(
    "HGC3DClusterGenMatchSelector",
    src = cms.InputTag('hgcalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
    genSrc = cms.InputTag('genParticles'),
    dR = cms.double(0.3)
)
