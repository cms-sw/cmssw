import FWCore.ParameterSet.Config as cms

mixSimHits = cms.PSet(
    input = cms.VInputTag(cms.InputTag("g4SimHits","MuonCSCHits"), cms.InputTag("g4SimHits","MuonDTHits"), cms.InputTag("g4SimHits","MuonRPCHits"), cms.InputTag("g4SimHits","TrackerHits")),
    type = cms.string('PSimHit'),
    subdets = cms.vstring('MuonCSCHits', 
        'MuonDTHits', 
        'MuonRPCHits', 
        'TrackerHits'),
    crossingFrames = cms.untracked.vstring('MuonCSCHits', 
        'MuonDTHits', 
        'MuonRPCHits', 
        'TrackerHits')
)
mixCaloHits = cms.PSet(
    input = cms.VInputTag(cms.InputTag("g4SimHits","EcalHitsEB"), cms.InputTag("g4SimHits","EcalHitsEE"), cms.InputTag("g4SimHits","EcalHitsES"), cms.InputTag("g4SimHits","HcalHits")),
    type = cms.string('PCaloHit'),
    subdets = cms.vstring('EcalHitsEB', 
        'EcalHitsEE', 
        'EcalHitsES', 
        'HcalHits'),
    crossingFrames = cms.untracked.vstring('EcalHitsEB', 
        'EcalHitsEE', 
        'EcalHitsES', # keep only ES and remove the others?
        'HcalHits')
)
mixSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("g4SimHits")),
    makeCrossingFrame = cms.untracked.bool(True),
    type = cms.string('SimTrack')
)
mixMuonSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("g4SimHits","MuonSimTracks")),
    makeCrossingFrame = cms.untracked.bool(True),
    type = cms.string('SimTrack')
)
mixSimVertices = cms.PSet(
    input = cms.VInputTag(cms.InputTag("g4SimHits")),
    makeCrossingFrame = cms.untracked.bool(True),
    type = cms.string('SimVertex')
)
mixHepMCProducts = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(True),
    input = cms.VInputTag(cms.InputTag("generator")),
    type = cms.string('HepMCProduct')
)

