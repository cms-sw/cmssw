# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

#AOD content
RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_conversionStepTracks_*_*',
        'keep recoTracks_beamhaloTracks_*_*',
        'keep recoTracks_ctfPixelLess_*_*', 
        'keep *_dedxHarmonic2_*_*',
        'keep *_dedxPixelHarmonic2_*_*',
        'keep *_dedxHitInfo_*_*',
        'keep *_trackExtrapolator_*_*',
        'keep *_generalTracks_MVAValues_*',
        'keep *_generalTracks_MVAVals_*'
    )
)
#HI-specific products: needed in AOD, propagate to more inclusive tiers as well
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify( RecoTrackerAOD.outputCommands, 
                   func=lambda outputCommands: outputCommands.extend(['keep recoTracks_hiConformalPixelTracks_*_*'])
)
#RECO content
RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoTrackExtras_generalTracks_*_*',
        'keep TrackingRecHitsOwned_generalTracks_*_*',
        'keep TrackingRecHitsOwned_extraFromSeeds_*_*',
        'keep uints_extraFromSeeds_*_*',                                   
        'keep recoTrackExtras_beamhaloTracks_*_*', 
        'keep TrackingRecHitsOwned_beamhaloTracks_*_*',
        'keep recoTrackExtras_conversionStepTracks_*_*', 
        'keep TrackingRecHitsOwned_conversionStepTracks_*_*',
        'keep *_ctfPixelLess_*_*', 
        'keep *_dedxTruncated40_*_*'
    )
)
RecoTrackerRECO.outputCommands.extend(RecoTrackerAOD.outputCommands)
pp_on_AA.toModify( RecoTrackerRECO.outputCommands, 
                   func=lambda outputCommands: outputCommands.extend([
                       'keep recoTrackExtras_hiConformalPixelTracks_*_*',
                       'keep TrackingRecHitsOwned_hiConformalPixelTracks_*_*'
                   ])
)
#Full Event content 
RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoTrackerFEVT.outputCommands.extend(RecoTrackerRECO.outputCommands)
