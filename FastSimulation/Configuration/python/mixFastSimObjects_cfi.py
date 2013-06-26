import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.CommonInputs_cff import * # temporary, just to select the MixingMode

if (MixingMode==2):
    mixSimHits = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits","MuonCSCHits"), cms.InputTag("g4SimHits","MuonDTHits"), cms.InputTag("g4SimHits","MuonRPCHits"), cms.InputTag("g4SimHits","TrackerHits")),
        type = cms.string('PSimHit'),
        subdets = cms.vstring('MuonCSCHits', 
                              'MuonDTHits', 
                              'MuonRPCHits', 
                              'TrackerHits'),
        crossingFrames = cms.untracked.vstring('MuonCSCHits', 
                                               'MuonDTHits', 
                                               'MuonRPCHits')
        )
    mixCaloHits = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits","EcalHitsEB"), cms.InputTag("g4SimHits","EcalHitsEE"), cms.InputTag("g4SimHits","EcalHitsES"), cms.InputTag("g4SimHits","HcalHits")),
        type = cms.string('PCaloHit'),
        subdets = cms.vstring('EcalHitsEB', 
                              'EcalHitsEE', 
                              'EcalHitsES', 
                              'HcalHits'),
        crossingFrames = cms.untracked.vstring()
        )
    mixSimTracks = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(False),
        type = cms.string('SimTrack')
        )
    mixMuonSimTracks = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits","MuonSimTracks")),
        makeCrossingFrame = cms.untracked.bool(False),
        type = cms.string('SimTrack')
        )
    mixSimVertices = cms.PSet(
        input = cms.VInputTag(cms.InputTag("g4SimHits")),
        makeCrossingFrame = cms.untracked.bool(False),
        type = cms.string('SimVertex')
        )
    mixReconstructedTracks = cms.PSet(
        input = cms.VInputTag(cms.InputTag("generalTracks")),
        type = cms.string('RecoTrack')
        )
else:
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
                                               'EcalHitsES', # keep only ES and remove the others?
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

