import FWCore.ParameterSet.Config as cms

######### This file becomes deprecated now that the TrackingParticles are produced by the MixingModule

#
# Cloned from SimGeneral.TrackingAnalysis.trackingParticles_cfi
# to allow different tracking particles for tracker and muon
# validation (until FastSim gets able to deal with a single
# trackingParticle collection)
#

mergedtruth = cms.EDProducer("TrackingTruthProducer",

    mixLabel = cms.string('mix'), # note: to become mixHits when the new sequences with mixing at SIM and RECO levels will be default
    simHitLabel = cms.string('famosSimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    volumeZ = cms.double(3000.0),
    mergedBremsstrahlung = cms.bool(True),
    removeDeadModules = cms.bool(False),

    HepMCDataLabels = cms.vstring('VtxSmeared', 
        'generator', 
        'PythiaSource', 
        'source'
    ),

    useMultipleHepMCLabels = cms.bool(False),

    simHitCollections = cms.PSet(
       tracker = cms.vstring("famosSimHitsTrackerHits"),
#       muon = cms.vstring (
#           'MuonSimHitsMuonDTHits',
#           'MuonSimHitsMuonCSCHits',
#           'MuonSimHitsMuonRPCHits'            
#       )
    )
)

mergedtruthMuon = cms.EDProducer("TrackingTruthProducer",

    mixLabel = cms.string('mix'), # note: to become mixHits when the new sequences with mixing at SIM and RECO levels will be default
    simHitLabel = cms.string('famosSimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    volumeZ = cms.double(3000.0),
    mergedBremsstrahlung = cms.bool(True),
    removeDeadModules = cms.bool(False),

    HepMCDataLabels = cms.vstring('VtxSmeared', 
        'generator', 
        'PythiaSource', 
        'source'
    ),

    useMultipleHepMCLabels = cms.bool(False),

    simHitCollections = cms.PSet(
       tracker = cms.vstring("famosSimHitsTrackerHits"),
       muon = cms.vstring (
           'MuonSimHitsMuonDTHits',
           'MuonSimHitsMuonCSCHits',
           'MuonSimHitsMuonRPCHits'            
       )
    )
)

trackingParticles = cms.Sequence(mergedtruth+mergedtruthMuon)

