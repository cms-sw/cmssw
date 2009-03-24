import FWCore.ParameterSet.Config as cms

# Tracking particle module
from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *
mergedtruth.TrackerHitLabels = ['famosSimHitsTrackerHits']
mergedtruth.simHitLabel = 'famosSimHits'

from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.TrackValidation_fastsim_cff import *
from Validation.RecoMuon.muonValidation_cff import *

globalValidation = cms.Sequence(trackingParticles+trackingTruthValid+tracksValidation)
