# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTrackExtras_generalTracks_*_*',
        'keep TrackingRecHitsOwned_generalTracks_*_*',
        'keep *_generalTracks_MVAValues_*',
        'keep TrackingRecHitsOwned_extraFromSeeds_*_*',
        'keep uints_extraFromSeeds_*_*',                                   
        'keep recoTracks_beamhaloTracks_*_*', 
        'keep recoTrackExtras_beamhaloTracks_*_*', 
        'keep TrackingRecHitsOwned_beamhaloTracks_*_*',
        'keep recoTracks_conversionStepTracks_*_*', 
        'keep recoTrackExtras_conversionStepTracks_*_*', 
        'keep TrackingRecHitsOwned_conversionStepTracks_*_*',
        'keep *_ctfPixelLess_*_*', 
        'keep *_dedxTruncated40_*_*',
        'keep *_dedxHitInfo_*_*',
        'keep *_dedxHarmonic2_*_*',
        'keep *_trackExtrapolator_*_*',
        'keep recoTracks_cosmicDCTracks_*_*',
        'keep recoTrackExtras_cosmicDCTracks_*_*',
        'keep TrackingRecHitsOwned_cosmicDCTracks_*_*',
     )
)
#RECO content
RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTrackExtras_generalTracks_*_*',
        'keep TrackingRecHitsOwned_generalTracks_*_*',
        'keep *_generalTracks_MVAValues_*',
        'keep *_generalTracks_MVAVals_*',
        'keep TrackingRecHitsOwned_extraFromSeeds_*_*',
        'keep uints_extraFromSeeds_*_*',                                   
        'keep recoTracks_beamhaloTracks_*_*', 
        'keep recoTrackExtras_beamhaloTracks_*_*', 
        'keep TrackingRecHitsOwned_beamhaloTracks_*_*',
        'keep recoTracks_conversionStepTracks_*_*', 
        'keep recoTrackExtras_conversionStepTracks_*_*', 
        'keep TrackingRecHitsOwned_conversionStepTracks_*_*',
        'keep *_ctfPixelLess_*_*', 
        'keep *_dedxTruncated40_*_*',
        'keep *_dedxHitInfo_*_*',
        'keep *_dedxHarmonic2_*_*',
        'keep *_trackExtrapolator_*_*',
        'keep recoTracks_cosmicDCTracks_*_*',
        'keep recoTrackExtras_cosmicDCTracks_*_*',
        'keep TrackingRecHitsOwned_cosmicDCTracks_*_*',
    )
)
#AOD content
RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_conversionStepTracks_*_*',
        'keep recoTracks_beamhaloTracks_*_*',
        'keep recoTracks_ctfPixelLess_*_*', 
        'keep *_dedxHarmonic2_*_*',
        'keep *_dedxHitInfo_*_*',
        'keep *_trackExtrapolator_*_*',
        'keep *_generalTracks_MVAValues_*',
        'keep *_generalTracks_MVAVals_*'
    )
)

