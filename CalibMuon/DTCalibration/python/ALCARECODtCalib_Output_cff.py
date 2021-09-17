import FWCore.ParameterSet.Config as cms

# output block for alcastream Electron
OutALCARECODtCalib_noDrop = cms.PSet(
    # put this if you have a filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECODtCalib')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DSegmentsNoWire_*_*',
        'keep *_muonDTDigis_*_*', 
        'keep *_dttfDigis_*_*',
        'keep *_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep recoMuons_muons_*_*',
        #'keep booledmValueMap_muid*_*_*')
        'keep booledmValueMap_muidAllArbitrated_*_*',
        'keep booledmValueMap_muidGMStaChiCompatibility_*_*',
        'keep booledmValueMap_muidGMTkChiCompatibility_*_*',
        'keep booledmValueMap_muidGMTkKinkTight_*_*',
        'keep booledmValueMap_muidGlobalMuonPromptTight_*_*',
        'keep booledmValueMap_muidRPCMuLoose_*_*',
        'keep booledmValueMap_muidTM2DCompatibilityLoose_*_*',
        'keep booledmValueMap_muidTM2DCompatibilityTight_*_*',
        'keep booledmValueMap_muidTMLastStationAngLoose_*_*',
        'keep booledmValueMap_muidTMLastStationAngTight_*_*',
        'keep booledmValueMap_muidTMLastStationLoose_*_*',
        'keep booledmValueMap_muidTMLastStationOptimizedLowPtLoose_*_*',
        'keep booledmValueMap_muidTMLastStationOptimizedLowPtTight_*_*',
        'keep booledmValueMap_muidTMLastStationTight_*_*',
        'keep booledmValueMap_muidTMOneStationAngLoose_*_*',
        'keep booledmValueMap_muidTMOneStationAngTight_*_*',
        'keep booledmValueMap_muidTMOneStationLoose_*_*',
        'keep booledmValueMap_muidTMOneStationTight_*_*',
        'keep booledmValueMap_muidTrackerMuonArbitrated_*_*')
)


import copy
OutALCARECODtCalib = copy.deepcopy(OutALCARECODtCalib_noDrop)
OutALCARECODtCalib.outputCommands.insert(0, "drop *")

## customizations for the pp_on_AA eras
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

OutALCARECODtCalibHI = copy.deepcopy(OutALCARECODtCalib_noDrop)
OutALCARECODtCalibHI.outputCommands.insert(0, "drop *")
OutALCARECODtCalibHI.outputCommands.append("keep *_offlinePrimaryVertices__*")
OutALCARECODtCalibHI.outputCommands.append("keep *_offlinePrimaryVerticesWithBS_*_*")
OutALCARECODtCalibHI.outputCommands.append("keep *_offlinePrimaryVerticesFromCosmicTracks_*_*")

#Specify to use HI output for the pp_on_AA eras
pp_on_AA.toReplaceWith(OutALCARECODtCalib,OutALCARECODtCalibHI)
