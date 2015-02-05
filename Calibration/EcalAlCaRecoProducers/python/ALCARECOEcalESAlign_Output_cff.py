import FWCore.ParameterSet.Config as cms


OutALCARECOEcalESAlign_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOEcalESAlign')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoBeamSpot_offlineBeamSpot_*_*',

        'keep *_ecalPreshowerDigis_*_*',
        'keep *_ecalAlCaESAlignTrackReducer_*_*',

        # 'keep recoTracks_ecalAlCaESAlignTrackReducer_*_*',
        # 'keep recoTrackExtras_ecalAlCaESAlignTrackReducer_*_*',
        # 'keep TrackingRecHitsOwned_ecalAlCaESAlignTrackReducer_*_*',
        # 'keep SiStripClusteredmNewDetSetVector_ecalAlCaESAlignTrackReducer_*_*',
        # 'keep SiPixelClusteredmNewDetSetVector_ecalAlCaESAlignTrackReducer_*_*',

        # 'keep ESDigiCollection_ecalPreshowerDigis_*_*',
        # 'keep ESDCCHeaderBlocksSorted_ecalPreshowerDigis_*_*',
        # 'keep ESKCHIPBlocksSorted_ecalPreshowerDigis_*_*',

        )
)


import copy
OutALCARECOEcalESAlign=copy.deepcopy(OutALCARECOEcalESAlign_noDrop)
OutALCARECOEcalESAlign.outputCommands.insert(0,"drop *")
