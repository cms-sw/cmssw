import FWCore.ParameterSet.Config as cms

# output block for alcastream
OutALCARECODtCalibCosmics_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECODtCalibCosmics')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DSegmentsNoWire_*_*',
        'keep *_muonDTDigis_*_*', 
        'keep *_dttfDigis_*_*',
        'keep *_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep recoTracks_cosmicMuons_*_*',
        'keep recoMuons_muons_*_*',
        #'keep booledmValueMap_muid*_*_*'
        'keep booledmValueMap muidAllArbitrated_*_*',
        'keep booledmValueMap muidGMStaChiCompatibility_*_*',
        'keep booledmValueMap muidGMTkChiCompatibility_*_*',
        'keep booledmValueMap muidGMTkKinkTight_*_*',
        'keep booledmValueMap muidGlobalMuonPromptTight_*_*',
        'keep booledmValueMap muidRPCMuLoose_*_*',
        'keep booledmValueMap muidTM2DCompatibilityLoose_*_*',
        'keep booledmValueMap muidTM2DCompatibilityTight_*_*',
        'keep booledmValueMap muidTMLastStationAngLoose_*_*',
        'keep booledmValueMap muidTMLastStationAngTight_*_*',
        'keep booledmValueMap muidTMLastStationLoose_*_*',
        'keep booledmValueMap muidTMLastStationOptimizedLowPtLoose_*_*',
        'keep booledmValueMap muidTMLastStationOptimizedLowPtTight_*_*',
        'keep booledmValueMap muidTMLastStationTight_*_*',
        'keep booledmValueMap muidTMOneStationAngLoose_*_*',
        'keep booledmValueMap muidTMOneStationAngTight_*_*',
        'keep booledmValueMap muidTMOneStationLoose_*_*',
        'keep booledmValueMap muidTMOneStationTight_*_*',
        'keep booledmValueMap muidTrackerMuonArbitrated_*_*')

)


import copy
OutALCARECODtCalibCosmics = copy.deepcopy(OutALCARECODtCalibCosmics_noDrop)
OutALCARECODtCalibCosmics.outputCommands.insert(0, "drop *")
