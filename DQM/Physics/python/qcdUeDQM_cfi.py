# $Id: qcdUeDQM_cfi.py,v 1.2 2010/04/16 12:52:59 olzem Exp $

import FWCore.ParameterSet.Config as cms
from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import *
from RecoJets.JetProducers.TracksForJets_cff import *
#from RecoJets.JetProducers.sc5TrackJets_cfi import sisCone5TrackJets
from RecoJets.JetProducers.ic5TrackJets_cfi import iterativeCone5TrackJets
recoTrackJets   =cms.Sequence( trackWithVertexRefSelector + trackRefsForJets + iterativeCone5TrackJets)



QcdUeDQM = cms.EDAnalyzer("QcdUeDQM",
    hltTrgNames  = cms.untracked.vstring(
    'HLT_ZeroBias'
    'HLT_MinBiasEcal',
    'HLT_MinBiasHcal',
    'HLT_MinBiasPixel'
    'HLT_MinBiasEcal',
    'HLT_MinBiasBSC',
    'HLT_ZeroBiasPixel_SingleTrack',
    'HLT_MinBiasPixel_SingleTrack',
    'HLT_MinBiasPixel_DoubleTrack',
    'HLT_MinBiasPixel_DoubleIsoTrack5',
    'HLT_L1Jet6U',
    'HLT_L1Jet10U',
    'HLT_Jet15U'
    ),
    caloJetTag = cms.untracked.InputTag("iterativeCone5CaloJets"),
    chargedJetTag = cms.untracked.InputTag("iterativeCone5TrackJets"),
    trackTag = cms.untracked.InputTag("generalTracks"),
    vtxTag = cms.untracked.InputTag("offlinePrimaryVertices"),                            
    beamSpotTag = cms.InputTag("offlineBeamSpot"),
    hltTrgResults = cms.untracked.string("TriggerResults"), 
    verbose = cms.untracked.int32(3),
    maxChi2 = cms.double(10000.0),
    tip = cms.double(5),                     #d0/sigmad0
    minRapidity = cms.double(-2.0),
    lip = cms.double(5),                     #dz/sigmadz
    ptMin = cms.double(0.5),
    maxRapidity = cms.double(2.0),
    vtxntk = cms.double(3),                  #selection evets with vertex recostructed with at least 3 tracks
    pxlLayerMinCut = cms.double(0),          #selection tracks with at least 2 hits in the pixel
    requirePIX1 = cms.bool(False),           #selection tracks with a hit a first layer of the pixel barrel or endcap 
    quality = cms.vstring('highPurity'),
    algorithm = cms.vstring(),
    minHit = cms.int32(3),                   #selection tracks with at least 4 layers crossed
    min3DHit = cms.int32(0),
    diffvtxbs =cms.double(15.),              #change selection from 10 to 15 
    ptErr_pt = cms.double(0.05),                #5%  
    bsuse = cms.bool(False),
    allowTriplets = cms.bool(False),            #change UE3 selection
    bsPos = cms.double(0) 
)
