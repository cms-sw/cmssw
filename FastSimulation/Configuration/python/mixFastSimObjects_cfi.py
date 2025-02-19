import FWCore.ParameterSet.Config as cms

mixSimHits = cms.PSet(
    input = cms.VInputTag(cms.InputTag("MuonSimHits","MuonCSCHits"), cms.InputTag("MuonSimHits","MuonDTHits"), cms.InputTag("MuonSimHits","MuonRPCHits"), cms.InputTag("famosSimHits","TrackerHits")),
    type = cms.string('PSimHit'),
    subdets = cms.vstring('MuonCSCHits', 
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
        'HcalHits')
)
mixSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits")),
    type = cms.string('SimTrack')
)
mixMuonSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits","MuonSimTracks")),
    type = cms.string('SimTrack')
)
mixSimVertices = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits")),
    type = cms.string('SimVertex')
)
mixHepMCProducts = cms.PSet(
    input = cms.VInputTag(cms.InputTag("generator")),
    type = cms.string('HepMCProduct')
)

