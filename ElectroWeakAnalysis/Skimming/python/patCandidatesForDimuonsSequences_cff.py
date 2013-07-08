import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# PAT TRACKS

# before pat: conversion to track candidates for pat; isolation 
#from PhysicsTools.PatAlgos.recoLayer0.genericTrackCandidates_cff import *
from ElectroWeakAnalysis.Skimming.patAODTrackCandSequence_cff import *
patAODTrackCands.cut = 'pt > 10.'

# before pat: MC match
#from PhysicsTools.PatAlgos.mcMatchLayer0.trackMuMatch_cfi import *
from ElectroWeakAnalysis.Skimming.trackMuMatch_cfi import *
trackMuMatch.maxDeltaR = 0.15
trackMuMatch.maxDPtRel = 1.0
trackMuMatch.resolveAmbiguities = True
trackMuMatch.resolveByMatchQuality = True

# pat tracks
from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import patGenericParticles

allPatTracks = patGenericParticles.clone(
    src = cms.InputTag("patAODTrackCands"),
    # isolation configurables
    userIsolation = cms.PSet(
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
selectedPatTracks.cut = 'pt > 10.'

# PAT MUONS

# before pat: MC match
from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import *
muonMatch.maxDeltaR = 0.15
muonMatch.maxDPtRel = 1.0
muonMatch.resolveAmbiguities = True
muonMatch.resolveByMatchQuality = True

# pat muons
# needed starting from 3_6_1
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
#
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *
patMuons.isoDeposits = cms.PSet(
        tracker = cms.InputTag("muIsoDepositTk"),
        ecal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
        hcal    = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
)
patMuons.userIsolation = cms.PSet(
        hcal = cms.PSet(
            src = cms.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
            deltaR = cms.double(0.3)
        ),
        tracker = cms.PSet(
            veto = cms.double(0.015),
            src = cms.InputTag("muIsoDepositTk"),
            deltaR = cms.double(0.3),
            threshold = cms.double(1.5)
            ),
        ecal = cms.PSet(
            src = cms.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
            deltaR = cms.double(0.3)
        )
    )

from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
selectedPatMuons.cut = 'pt > 0. & abs(eta) < 100.0'

# trigger info
from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import *
# to access 8E29 menus
#patTrigger.triggerResults = cms.InputTag( "TriggerResults::HLT8E29" )
#patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT8E29" )
# to access 1E31 menus
patTrigger.triggerResults = cms.InputTag( "TriggerResults::HLT" )
patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT" )

muonTriggerMatchHLTMuons = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "selectedPatMuons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( 'TriggerMuon' ), # 'TriggerMuon' is the enum from trigger::TriggerObjectType for HLT muons
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_Mu9' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 1.0 ),
    maxDeltaR = cms.double( 0.2 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( True )
)

from PhysicsTools.PatAlgos.triggerLayer1.triggerEventProducer_cfi import *
#patTriggerEvent.patTriggerMatches  = [ "muonTriggerMatchHLTMu9" ]
patTriggerEvent.patTriggerMatches  = cms.VInputTag( "muonTriggerMatchHLTMuons" )
#patTriggerEvent.patTriggerMatches  = cms.VInputTag( "muonTriggerMatchHLTMu9" )

patTriggerSequence = cms.Sequence(
    patTrigger *
    muonTriggerMatchHLTMuons *
#    muonTriggerMatchHLTMu9 *
    patTriggerEvent
)

selectedPatMuonsTriggerMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag( "selectedPatMuons" ),
    matches = cms.VInputTag( "muonTriggerMatchHLTMuons" )
)

muonTriggerMatchEmbedder = cms.Sequence(
    selectedPatMuonsTriggerMatch
)

# pat sequences

beforePatTracks = cms.Sequence(
    patAODTrackCandSequence *
    trackMuMatch
)

beforePatMuons = cms.Sequence(
    muonMatch
)

beforePat = cms.Sequence(
    beforePatTracks +
    beforePatMuons
)

patCandsSequence = cms.Sequence(
    patMuons *
    selectedPatMuons *
    allPatTracks *
    selectedPatTracks
)

goodMuonRecoForDimuon = cms.Sequence(
    beforePat *
    patCandsSequence *
    patTriggerSequence *
    muonTriggerMatchEmbedder
)

