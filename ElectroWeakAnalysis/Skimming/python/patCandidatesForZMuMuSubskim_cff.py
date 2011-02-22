import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
#####################################################
# PAT muons and tracks for ZMuMu subskim: no MC match
#####################################################

# PAT TRACKS 

# before pat: conversion to track candidates for pat; isolation 
from ElectroWeakAnalysis.Skimming.patAODTrackCandSequence_cff import *
patAODTrackCands.cut = 'pt > 15.'

# pat tracks
from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import patGenericParticles

allPatTracks = patGenericParticles.clone(
    src = cms.InputTag("patAODTrackCands"),
    embedTrack = cms.bool(True),
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
    addGenMatch = cms.bool(False)
)

from PhysicsTools.PatAlgos.selectionLayer1.trackSelector_cfi import *
selectedPatTracks.cut = 'pt > 15. & track.dxy()<1.0'

# PAT MUONS

# before pat: Merge CaloMuons into the collection of reco::Muons
# Starting from 3_4_X a special recipe is needed for CaloMuons merging
# Uncomment the following lines and follow the recipe in:
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIsolation#To_remake_IsoDeposits_in_CMSSW_3

#from RecoMuon.MuonIdentification.calomuons_cfi import calomuons;
#muons = cms.EDProducer("CaloMuonMerger",
#    muons = cms.InputTag("muons"), # half-dirty thing. it works aslong as we're the first module using muons in the path
#    caloMuons = cms.InputTag("calomuons"),
#    minCaloCompatibility = calomuons.minCaloCompatibility)

## And re-make isolation, as we can't use the one in AOD because our collection is different
#import RecoMuon.MuonIsolationProducers.muIsolation_cff

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

patMuons.addGenMatch = cms.bool(False)
patMuons.embedTrack = cms.bool(True)
patMuons.embedCombinedMuon = cms.bool(True)
patMuons.embedStandAloneMuon = cms.bool(True)
patMuons.embedPickyMuon = cms.bool(False)
patMuons.embedTpfmsMuon = cms.bool(False)
patMuons.embedPFCandidate = cms.bool(False)

from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
selectedPatMuons.cut = 'pt > 15. & abs(eta) < 100.0 & ( (isGlobalMuon==1  & innerTrack.dxy()<1.0)  | ((isTrackerMuon==1  & innerTrack.dxy()<1.0) | (isStandAloneMuon==1  & outerTrack.dxy()<1.0) ))'

# trigger info
from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import *
# to access 8E29 menus
#patTrigger.triggerResults = cms.InputTag( "TriggerResults::HLT8E29" )
#patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT8E29" )
# to access 1E31 menus
patTrigger.triggerResults = cms.InputTag( "TriggerResults::HLT" )
patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT" )

muonTriggerMatchHLTMuons = cms.EDProducer( "PATTriggerMatcherDRDPtLessByR",
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
    resolveByMatchQuality = cms.bool( False )
)

from PhysicsTools.PatAlgos.triggerLayer1.triggerEventProducer_cfi import *
patTriggerEvent.patTriggerMatches  = cms.VInputTag( "muonTriggerMatchHLTMuons" )

patTriggerSequence = cms.Sequence(
    patTrigger *
    muonTriggerMatchHLTMuons *
    patTriggerEvent
)

selectedPatMuonsTriggerMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag( "selectedPatMuons" ),
    matches = cms.VInputTag( "muonTriggerMatchHLTMuons" )
)

muonTriggerMatchEmbedder = cms.Sequence(
    selectedPatMuonsTriggerMatch
)

# uncomment in case of CaloMuons merging
#beforePatMuons = cms.Sequence(
#    muons *
#    muIsolation
#)

beforePatTracks = cms.Sequence(
    patAODTrackCandSequence 
)

beforePat = cms.Sequence(
# uncomment in case of CaloMuons merging
#    beforePatMuons *
    beforePatTracks
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

