import FWCore.ParameterSet.Config as cms

theMixObjects = cms.PSet(
    mixCH = cms.PSet(
        crossingFrames = cms.untracked.vstring(),
        input = cms.VInputTag(
            cms.InputTag("g4SimHits","CastorFI"), cms.InputTag("g4SimHits","EcalHitsEB"), cms.InputTag("g4SimHits","EcalHitsEE"), cms.InputTag("g4SimHits","EcalHitsES"), cms.InputTag("g4SimHits","HcalHits"),
            cms.InputTag("g4SimHits","ZDCHITS"), cms.InputTag("g4SimHits","HGCHitsEE"), cms.InputTag("g4SimHits","HGCHitsHEfront"), cms.InputTag("g4SimHits","HGCHitsHEback")
        ),
        subdets = cms.vstring(
            'CastorFI',
            'EcalHitsEB',
            'EcalHitsEE',
            'EcalHitsES',
            'HcalHits',
            'ZDCHITS',
            'HGCHitsEE',
            'HGCHitsHEfront',
            'HGCHitsHEback'
        ),
        type = cms.string('PCaloHit')
    ),
    mixHepMC = cms.PSet(
        input = cms.VInputTag(cms.InputTag("generatorSmeared"), cms.InputTag("generator")),
        makeCrossingFrame = cms.untracked.bool(True),
        type = cms.string('HepMCProduct')
    ),
    mixSH = cms.PSet(
        crossingFrames = cms.untracked.vstring(
            'MuonCSCHits',
            'MuonDTHits',
            'MuonRPCHits',
            'MuonGEMHits',
            'MuonME0Hits',
            'FastTimerHitsBarrel',
            'FastTimerHitsEndcap'
        ),
        input = cms.VInputTag(
            cms.InputTag("g4SimHits","MuonCSCHits"), cms.InputTag("g4SimHits","MuonDTHits"), cms.InputTag("g4SimHits","MuonRPCHits"), cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"), cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
            cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"), cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"), cms.InputTag("g4SimHits","TrackerHitsTECHighTof"), cms.InputTag("g4SimHits","TrackerHitsTECLowTof"), cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"),
            cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"), cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"), cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"), cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"), cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"),
            cms.InputTag("g4SimHits","MuonGEMHits"), cms.InputTag("g4SimHits","MuonME0Hits"), cms.InputTag("g4SimHits","FastTimerHitsBarrel"), cms.InputTag("g4SimHits","FastTimerHitsEndcap")
        ),
        pcrossingFrames = cms.untracked.vstring(),
        subdets = cms.vstring(
            'MuonCSCHits',
            'MuonDTHits',
            'MuonRPCHits',
            'TrackerHitsPixelBarrelHighTof',
            'TrackerHitsPixelBarrelLowTof',
            'TrackerHitsPixelEndcapHighTof',
            'TrackerHitsPixelEndcapLowTof',
            'TrackerHitsTECHighTof',
            'TrackerHitsTECLowTof',
            'TrackerHitsTIBHighTof',
            'TrackerHitsTIBLowTof',
            'TrackerHitsTIDHighTof',
            'TrackerHitsTIDLowTof',
            'TrackerHitsTOBHighTof',
            'TrackerHitsTOBLowTof',
            'MuonGEMHits',
            'MuonME0Hits',
            'FastTimerHitsBarrel',
            'FastTimerHitsEndcap'
        ),
        type = cms.string('PSimHit')
    ),
    mixTracks = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(False),
        type = cms.string('SimTrack')
    ),
    mixVertices = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(False),
        type = cms.string('SimVertex')
    )
)