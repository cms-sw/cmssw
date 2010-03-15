# $Id: qcdLowPtDQM_cfi.py,v 1.7 2009/11/23 10:06:49 loizides Exp $

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
    ),
    caloJetTag = cms.untracked.InputTag("iterativeCone5CaloJets"),
    chargedJetTag = cms.untracked.InputTag("iterativeCone5TrackJets"),
    trackTag = cms.untracked.InputTag("generalTracks"),
    vtxTag = cms.untracked.InputTag("offlinePrimaryVertices"),                            
    hltTrgResults = cms.untracked.string("TriggerResults"), 
    verbose = cms.untracked.int32(3)
)
