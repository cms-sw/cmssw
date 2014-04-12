import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *

triggerMatcherToHLTDebug = cms.EDProducer("TriggerMatcherToHLTDebug",
    tags = cms.InputTag("muons"), 
    l1s  = cms.InputTag("hltL1extraParticles"),
    L2Muons_Collection=cms.InputTag("hltL2MuonCandidates"),
    L2Seeds_Collection=cms.InputTag("hltL2MuonSeeds"),
    L3Seeds_Collection=cms.InputTag("hltL3TrajectorySeed"),
    L3TkTracks_Collection=cms.InputTag("hltL3TkTracksFromL2"),
    L3Muons_Collection=cms.InputTag("hltL3MuonCandidates"), 
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),

    l1matcherConfig = cms.PSet(
        #useTrack = cms.string('global'),
        #useState = cms.string('outermost'),
        useTrack = cms.string("tracker"),  # 'none' to use Candidate P4; or 'tracker', 'muon', 'global'
        useState = cms.string("atVertex"), # 'innermost' and 'outermost' require the TrackExtra
        useSimpleGeometry = cms.bool(True),
    ),
    deltaR  = cms.double(0.3),

    #FilteringL1
    MinL1Quality = cms.int32( 1 ),

    #associationMap                                      
    SeedMapTag = cms.InputTag( "hltL2Muons" ),



    #FilteringL2
    #PreviousCandTag_L2 = cms.InputTag( "hltL1SingleMu3L1Filtered0" ),
    MinN_L2 = cms.int32( 1 ),
    MaxEta_L2 = cms.double( 2.5 ),
    MinNhits_L2 = cms.int32( 0 ),
    MaxDr_L2 = cms.double( 9999.0 ),
    MaxDz_L2 = cms.double( 9999.0 ),
    MinPt_L2 = cms.double( 7.0 ),
    NSigmaPt_L2 = cms.double( 0.0 ),

    #PreviousCandTag_L3 = cms.InputTag( "hltSingleMu5L2Filtered4" ),
    MinN_L3 = cms.int32( 1 ),
    MaxEta_L3 = cms.double( 2.5 ),
    MinNhits_L3 = cms.int32( 0 ),
    MaxDr_L3 = cms.double( 2.0 ),
    MaxDz_L3 = cms.double( 9999.0 ),
    MinPt_L3 = cms.double( 9.0 ),
    NSigmaPt_L3 = cms.double( 0.0 ),
)

def addUserData(patMuonProducer, matcherLabel='triggerMatcherToHLTDebug'):
    patMuonProducer.userData.userInts.src += [
        cms.InputTag(matcherLabel,"propagatesToM2"),
        cms.InputTag(matcherLabel,"hasL1Particle"),
        cms.InputTag(matcherLabel,"hasL1Filtered"),
        cms.InputTag(matcherLabel,"hasL2Seed"),
        cms.InputTag(matcherLabel,"hasL2Muon"),
        cms.InputTag(matcherLabel,"hasL2MuonFiltered"),
        cms.InputTag(matcherLabel,"hasL3Seed"),
        cms.InputTag(matcherLabel,"hasL3Track"),
        cms.InputTag(matcherLabel,"hasL3Muon"),
        cms.InputTag(matcherLabel,"hasL3MuonFiltered"),
    ]
    patMuonProducer.userData.userCands.src += [
        cms.InputTag(matcherLabel,"l1Candidate"),
        cms.InputTag(matcherLabel,"l2Candidate"),
        cms.InputTag(matcherLabel,"l3Candidate"),
    ]

