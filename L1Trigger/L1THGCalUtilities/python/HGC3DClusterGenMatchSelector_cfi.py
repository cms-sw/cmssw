import FWCore.ParameterSet.Config as cms

l1tHGCal3DClusterGenMatchSelector = cms.EDProducer(
    "HGC3DClusterGenMatchSelector",
    src = cms.InputTag('l1tHGCalBackEndLayer2Producer:HGCalBackendLayer2Processor3DClustering'),
    genSrc = cms.InputTag('genParticles'),
    dR = cms.double(0.3)
)
