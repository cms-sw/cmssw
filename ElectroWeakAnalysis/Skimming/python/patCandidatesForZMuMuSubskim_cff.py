import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
#####################################################
# PAT muons and tracks for ZMuMu subskim: no MC match
#####################################################

# PAT TRACKS 

# before layer 1: conversion to track candidates for pat; isolation 
from ElectroWeakAnalysis.Skimming.patAODTrackCandSequence_cff import *
patAODTrackCands.cut = 'pt > 10.'

# layer 1 tracks
import PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi as genericpartproducer_cfi

allLayer1TrackCands = genericpartproducer_cfi.allLayer1GenericParticles.clone(
    src = cms.InputTag("patAODTrackCands"),
    embedTrack = cms.bool(True),
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
    addGenMatch = cms.bool(False)
)

from PhysicsTools.PatAlgos.selectionLayer1.trackSelector_cfi import *
selectedLayer1TrackCands.cut = 'pt > 10.'

# PAT MUONS

# before layer 1: Merge CaloMuons into the collection of reco::Muons
from RecoMuon.MuonIdentification.calomuons_cfi import calomuons;
muons = cms.EDProducer("CaloMuonMerger",
    muons = cms.InputTag("muons"), # half-dirty thing. it works aslong as we're the first module using muons in the path
    caloMuons = cms.InputTag("calomuons"),
    minCaloCompatibility = calomuons.minCaloCompatibility)

## And re-make isolation, as we can't use the one in AOD because our collection is different
import RecoMuon.MuonIsolationProducers.muIsolation_cff

# layer 1 muons
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *
allLayer1Muons.isolation.tracker = cms.PSet(
    veto = cms.double(0.015),
    src = cms.InputTag("muIsoDepositTk"),
    deltaR = cms.double(0.3),
    cut = cms.double(3.0),
    threshold = cms.double(1.5)
)
allLayer1Muons.addGenMatch = cms.bool(False)
allLayer1Muons.embedTrack = cms.bool(True)
allLayer1Muons.embedCombinedMuon = cms.bool(True)
allLayer1Muons.embedStandAloneMuon = cms.bool(True)
allLayer1Muons.embedPickyMuon = cms.bool(False)
allLayer1Muons.embedTpfmsMuon = cms.bool(False)
allLayer1Muons.embedPFCandidate = cms.bool(False)

from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
selectedLayer1Muons.cut = 'pt > 10. & abs(eta) < 100.0'

# trigger info
from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import *
# to access 8E29 menus
#patTrigger.triggerResults = cms.InputTag( "TriggerResults::HLT8E29" )
#patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT8E29" )
# to access 1E31 menus
patTrigger.triggerResults = cms.InputTag( "TriggerResults::HLT" )
patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT" )

muonTriggerMatchHLTMuons = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "selectedLayer1Muons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( 'TriggerMuon' ), # 'TriggerMuon' is the enum from trigger::TriggerObjectType for HLT muons
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( '*' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 1.0 ),
    maxDeltaR = cms.double( 0.2 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( False )
)

from PhysicsTools.PatAlgos.triggerLayer1.triggerEventProducer_cfi import *
patTriggerEvent.patTriggerMatches  = cms.VInputTag( "muonTriggerMatchHLTMuons" )

patTriggerSequence = cms.Sequence(
    patTrigger *
    muonTriggerMatchHLTMuons *
    patTriggerEvent
)

selectedLayer1MuonsTriggerMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag( "selectedLayer1Muons" ),
    matches = cms.VInputTag( "muonTriggerMatchHLTMuons" )
)

muonTriggerMatchEmbedder = cms.Sequence(
    selectedLayer1MuonsTriggerMatch
)

# pat sequences

beforeLayer1Muons = cms.Sequence(
    muons *
    muIsolation
)

beforeLayer1Tracks = cms.Sequence(
    patAODTrackCandSequence 
)

beforePatLayer1 = cms.Sequence(
    beforeLayer1Muons *
    beforeLayer1Tracks
)

patLayer1 = cms.Sequence(
    allLayer1Muons *
    selectedLayer1Muons *
    allLayer1TrackCands *
    selectedLayer1TrackCands
)

goodMuonRecoForDimuon = cms.Sequence(
    beforePatLayer1 *
    patLayer1 *
    patTriggerSequence *
    muonTriggerMatchEmbedder
)

