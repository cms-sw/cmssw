# cloned from SimGeneral/MixingModule/python/trackingTruthProducer_cfi.py

import FWCore.ParameterSet.Config as cms

trackingParticles = cms.PSet(
    accumulatorType = cms.string('TrackingTruthAccumulator'),
    createUnmergedCollection = cms.bool(True),
    createMergedBremsstrahlung = cms.bool(True),
    alwaysAddAncestors = cms.bool(True),
    maximumPreviousBunchCrossing = cms.uint32(9999),
    maximumSubsequentBunchCrossing = cms.uint32(9999),
    simHitCollections = cms.PSet(
        muon = cms.VInputTag( cms.InputTag('MuonSimHits','MuonDTHits'),
                       cms.InputTag('MuonSimHits','MuonCSCHits'),
                       cms.InputTag('MuonSimHits','MuonRPCHits') ),
        trackerAndPixel = cms.VInputTag( cms.InputTag('famosSimHits','TrackerHits') )
    ),
    simTrackCollection = cms.InputTag('famosSimHits'),
    simVertexCollection = cms.InputTag('famosSimHits'),
    genParticleCollection = cms.InputTag('genParticles'),
    removeDeadModules = cms.bool(False), # currently not implemented
    volumeRadius = cms.double(120.0),
    volumeZ = cms.double(300.0),
    ignoreTracksOutsideVolume = cms.bool(False),
    allowDifferentSimHitProcesses = cms.bool(True) # should be True for FastSim, False for FullSim
    )

# cloned from SimGeneral/MixingModule/python/trackingTruthProducerSelection_cfi.py

trackingParticles.select = cms.PSet(
        lipTP = cms.double(1000),
        chargedOnlyTP = cms.bool(True),
        pdgIdTP = cms.vint32(),
        signalOnlyTP = cms.bool(True),
        stableOnlyTP = cms.bool(True), # this is different from the standard setting for FullSim
        minRapidityTP = cms.double(-2.6),
        minHitTP = cms.int32(3),
        ptMinTP = cms.double(0.2),
        maxRapidityTP = cms.double(2.6),
        tipTP = cms.double(1000)
        )
