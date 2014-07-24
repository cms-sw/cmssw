# cloned from SimGeneral/MixingModule/python/trackingTruthProducer_cfi.py

import FWCore.ParameterSet.Config as cms
from SimGeneral.MixingModule.trackingTruthProducer_cfi import trackingParticles
from SimGeneral.MixingModule.trackingTruthProducerSelection_cfi import trackingParticlesSelection


trackingParticles.allowDifferentSimHitProcesses = True
trackingParticles.simHitCollections = cms.PSet(
        muon = cms.VInputTag( cms.InputTag('MuonSimHits','MuonDTHits'),
                       cms.InputTag('MuonSimHits','MuonCSCHits'),
                       cms.InputTag('MuonSimHits','MuonRPCHits') ),
        trackerAndPixel = cms.VInputTag( cms.InputTag('famosSimHits','TrackerHits') )
    )
trackingParticles.simTrackCollection = cms.InputTag('famosSimHits')
trackingParticles.simVertexCollection = cms.InputTag('famosSimHits')

trackingParticles.select = cms.PSet(trackingParticlesSelection)

