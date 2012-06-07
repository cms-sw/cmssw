import FWCore.ParameterSet.Config as cms

mixSimHits = cms.PSet(
    input = cms.VInputTag(cms.InputTag("MuonSimHits","MuonCSCHits"), cms.InputTag("MuonSimHits","MuonDTHits"), cms.InputTag("MuonSimHits","MuonRPCHits"), cms.InputTag("famosSimHits","TrackerHits")),
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
    input = cms.VInputTag(cms.InputTag("famosSimHits","EcalHitsEB"), cms.InputTag("famosSimHits","EcalHitsEE"), cms.InputTag("famosSimHits","EcalHitsES"), cms.InputTag("famosSimHits","HcalHits")),
    type = cms.string('PCaloHit'),
    subdets = cms.vstring('EcalHitsEB', 
        'EcalHitsEE', 
        'EcalHitsES', 
        'HcalHits'),
    crossingFrames = cms.untracked.vstring('EcalHitsEB', 
        'EcalHitsEE', 
        'EcalHitsES', 
        'HcalHits')
)
mixSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits")),
    makeCrossingFrame = cms.untracked.bool(True),
    type = cms.string('SimTrack')
)
mixMuonSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits","MuonSimTracks")),
    makeCrossingFrame = cms.untracked.bool(True),
    type = cms.string('SimTrack')
)
mixSimVertices = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits")),
    makeCrossingFrame = cms.untracked.bool(True),
    type = cms.string('SimVertex')
)
mixHepMCProducts = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(True),
    input = cms.VInputTag(cms.InputTag("generator")),
    type = cms.string('HepMCProduct')
)

