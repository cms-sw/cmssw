import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# PAT TRACKS (sequence compatible with PAT v2)

# before layer 1: conversion to track candidates for pat; isolation 
#from PhysicsTools.PatAlgos.recoLayer0.genericTrackCandidates_cff import *
from ElectroWeakAnalysis.ZReco.patAODTrackCandSequence_cff import *
patAODTrackCands.cut = 'pt > 10.'

# before layer 1: MC match
#from PhysicsTools.PatAlgos.mcMatchLayer0.trackMuMatch_cfi import *
from ElectroWeakAnalysis.ZReco.trackMuMatch_cfi import *
trackMuMatch.maxDeltaR = 0.15
trackMuMatch.maxDPtRel = 1.0
trackMuMatch.resolveAmbiguities = True
trackMuMatch.resolveByMatchQuality = True

# layer 1 tracks
import PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi as genericpartproducer_cfi

allLayer1TrackCands = genericpartproducer_cfi.allLayer1GenericParticles.clone(
    src = cms.InputTag("patAODTrackCands"),
    # isolation configurables
    isolation = cms.PSet(
      tracker = cms.PSet(
        veto = cms.double(0.015),
        src = cms.InputTag("patAODTrackIsoDepositCtfTk"),
        deltaR = cms.double(0.3),
        threshold = cms.double(1.5)
      ),
      ecal = cms.PSet(
        src = cms.InputTag("patAODTrackIsoDepositCalByAssociatorTowers","ecal"),
        deltaR = cms.double(0.3)
      ),
      hcal = cms.PSet(
        src = cms.InputTag("patAODTrackIsoDepositCalByAssociatorTowers","hcal"),
        deltaR = cms.double(0.3)
      ),
    ),
    isoDeposits = cms.PSet(
      tracker = cms.InputTag("patAODTrackIsoDepositCtfTk"),
      ecal = cms.InputTag("patAODTrackIsoDepositCalByAssociatorTowers","ecal"),
      hcal = cms.InputTag("patAODTrackIsoDepositCalByAssociatorTowers","hcal")
    ),
    addGenMatch = cms.bool(True),
    genParticleMatch = cms.InputTag("trackMuMatch")
)

from PhysicsTools.PatAlgos.selectionLayer1.trackSelector_cfi import *
selectedLayer1TrackCands.cut = 'pt > 10.'

# PAT MUONS (sequence compatible with PAT v2)

# before layer 1: MC match
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
muonMatch.maxDeltaR = 0.15
muonMatch.maxDPtRel = 1.0
muonMatch.resolveAmbiguities = True
muonMatch.resolveByMatchQuality = True

# layer 1 muons
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *
allLayer1Muons.isolation.tracker = cms.PSet(
    veto = cms.double(0.015),
    src = cms.InputTag("muIsoDepositTk"),
    deltaR = cms.double(0.3),
    cut = cms.double(3.0),
    threshold = cms.double(1.5)
)
#allLayer1Muons.addTrigMatch = cms.bool(False)

from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
selectedLayer1Muons.cut = 'pt > 0. & abs(eta) < 100.0'

# trigger info
from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import *

#muonTriggerMatchHLTMu15 = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
#    src     = cms.InputTag( "selectedLayer1Muons" ),
#    matched = cms.InputTag( "patTrigger" ),
#    andOr          = cms.bool( False ),
#    filterIdsEnum  = cms.vstring( '*' ),
#    filterIds      = cms.vint32( 0 ),
#    filterLabels   = cms.vstring( '*' ),
#    pathNames      = cms.vstring( 'HLT_Mu15' ),
#    collectionTags = cms.vstring( '*' ),
#    maxDPtRel = cms.double( 1.0 ),
#    maxDeltaR = cms.double( 0.5 ),
#    resolveAmbiguities    = cms.bool( True ),
#    resolveByMatchQuality = cms.bool( False )
#)

muonTriggerMatchHLTMuons = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "selectedLayer1Muons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( 'TriggerMuon' ), # 'TriggerMuon' is the enum from trigger::TriggerObjectType for HLT muons
    filterIds      = cms.vint32( 83 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( '*' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 1.0 ),
    maxDeltaR = cms.double( 0.2 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( False )
)

from PhysicsTools.PatAlgos.triggerLayer1.triggerEventProducer_cfi import *
#patTriggerEvent.patTriggerMatches  = [ "muonTriggerMatchHLTMu15" ]
patTriggerEvent.patTriggerMatches  = cms.VInputTag( "muonTriggerMatchHLTMuons" )
#patTriggerEvent.patTriggerMatches  = cms.VInputTag( "muonTriggerMatchHLTMu15" )

patTriggerSequence = cms.Sequence(
    patTrigger *
    muonTriggerMatchHLTMuons *
#    muonTriggerMatchHLTMu15 *
    patTriggerEvent
)

#selectedLayer1MuonsTriggerMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
#    src     = cms.InputTag( "selectedLayer1Muons" ),
#    matches = cms.VInputTag( "muonTriggerMatchHLTMuons" )
#)

#muonTriggerMatchEmbedder = cms.Sequence(
#    selectedLayer1MuonsTriggerMatch
#)

# pat sequences

beforeLayer1Tracks = cms.Sequence(
    patAODTrackCandSequence *
    trackMuMatch
)

beforeLayer1Muons = cms.Sequence(
    muonMatch # +
#    patTrigMatch
)

beforePatLayer1 = cms.Sequence(
    beforeLayer1Tracks +
    beforeLayer1Muons
)

patLayer1 = cms.Sequence(
    allLayer1Muons *
    selectedLayer1Muons *
#    cleanLayer1Muons *
    allLayer1TrackCands *
    selectedLayer1TrackCands
)

goodMuonRecoForDimuon = cms.Sequence(
    beforePatLayer1 *
    patLayer1 *
    patTriggerSequence # *
#    muonTriggerMatchEmbedder
)

