#######################
# this file is the FastSim equivalent of SimGeneral/MixingModule/python/mixObjects_cfi.py
# last reviewer:    Lukas Vanelderen
# last review data: Jan 20 2015
#######################

import FWCore.ParameterSet.Config as cms

mixSimHits = cms.PSet(
    input = cms.VInputTag(
        cms.InputTag("MuonSimHits","MuonCSCHits"), 
        cms.InputTag("MuonSimHits","MuonDTHits"), 
        cms.InputTag("MuonSimHits","MuonRPCHits"), 
        cms.InputTag("famosSimHits","TrackerHits")),
    type = cms.string('PSimHit'),
    subdets = cms.vstring(
        'MuonCSCHits', 
        'MuonDTHits', 
        'MuonRPCHits', 
        'TrackerHits'),
    # muon hits need crossingFrame
    crossingFrames = cms.untracked.vstring(
        'MuonCSCHits', 
        'MuonDTHits', 
        'MuonRPCHits', 
        'TrackerHits')
    )

mixCaloHits = cms.PSet(
    input = cms.VInputTag(
        cms.InputTag("famosSimHits","EcalHitsEB"), 
        cms.InputTag("famosSimHits","EcalHitsEE"), 
        cms.InputTag("famosSimHits","EcalHitsES"), 
        cms.InputTag("famosSimHits","HcalHits")),
    type = cms.string('PCaloHit'),
    subdets = cms.vstring(
        'EcalHitsEB', 
        'EcalHitsEE', 
        'EcalHitsES', 
        'HcalHits'),
    crossingFrames = cms.untracked.vstring()
    )

# fastsim mixes reconstructed tracks
mixReconstructedTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("generalTracksBeforeMixing")),
    type = cms.string('RecoTrack')
    )

mixSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits")),
    makeCrossingFrame = cms.untracked.bool(False),
    type = cms.string('SimTrack')
    )

# fastsim has separate SimTrack collection for muons
mixMuonSimTracks = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits","MuonSimTracks")),
    makeCrossingFrame = cms.untracked.bool(False),
    type = cms.string('SimTrack')
    )

mixSimVertices = cms.PSet(
    input = cms.VInputTag(cms.InputTag("famosSimHits")),
    makeCrossingFrame = cms.untracked.bool(False),
    type = cms.string('SimVertex')
    )
        
mixHepMCProducts = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(False),
    input = cms.VInputTag(cms.InputTag("generatorSmeared"),cms.InputTag("generator")),
    type = cms.string('HepMCProduct')
)

theMixObjects = cms.PSet(
    # as for FullSim
    mixCH = cms.PSet(mixCaloHits),
    mixTracks = cms.PSet(mixSimTracks),     # we stick to the confusing FullSim languate: tracks are actuall SimTracks
    mixVertices = cms.PSet(mixSimVertices), # same for SimVertices
    mixSH = cms.PSet(mixSimHits),
    mixHepMC = cms.PSet(mixHepMCProducts),
    # FastSim specific:
    mixMuonTracks = cms.PSet(mixMuonSimTracks),
    mixRecoTracks = cms.PSet(mixReconstructedTracks)
    )
