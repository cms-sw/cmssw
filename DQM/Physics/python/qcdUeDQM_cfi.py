# $Id: qcdUeDQM_cfi.py,v 1.4 2011/02/18 13:57:46 vlimant Exp $

import FWCore.ParameterSet.Config as cms


QcdUeDQM = cms.EDAnalyzer("QcdUeDQM",
    hltTrgNames  = cms.untracked.vstring(
    'HLT_ZeroBias',
    'HLT_ZeroBiasPixel_SingleTrack',
    'HLT_MinBiasPixel_SingleTrack',
    'HLT_L1Jet6U',
    'HLT_L1Jet10U',
    'HLT_Jet15U_v3'
    ),
    caloJetTag = cms.untracked.InputTag("ak5CaloJets"),
    chargedJetTag = cms.untracked.InputTag("ak5TrackJets"),
    trackTag = cms.untracked.InputTag("generalTracks"),
    vtxTag = cms.untracked.InputTag("offlinePrimaryVertices"),                            
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    hltTrgResults = cms.untracked.string("TriggerResults"), 
    verbose = cms.untracked.int32(3),
    maxChi2 = cms.double(10000.0),
    tip = cms.double(3),                     #d0/sigmad0
    minRapidity = cms.double(-2.0),
    lip = cms.double(3),                     #dz/sigmadz
    ptMin = cms.double(0.5),
    maxRapidity = cms.double(2.0),
    vtxntk = cms.double(4),                  #selection evets with vertex recostructed with at least 3 tracks
    pxlLayerMinCut = cms.double(0),          #selection tracks with at least 2 hits in the pixel
    requirePIX1 = cms.bool(False),           #selection tracks with a hit a first layer of the pixel barrel or endcap 
    quality = cms.vstring('highPurity'),
    algorithm = cms.vstring(),
    minHit = cms.int32(3),                   #selection tracks with at least 4 layers crossed
    min3DHit = cms.int32(0),
    diffvtxbs =cms.double(10.),              #change selection from 10 to 15 
    ptErr_pt = cms.double(0.05),                #5%  
    bsuse = cms.bool(False),
    allowTriplets = cms.bool(False),            #change UE3 selection
    bsPos = cms.double(0) ,

    hltProcNames = cms.untracked.vstring(['HLT'])
)
