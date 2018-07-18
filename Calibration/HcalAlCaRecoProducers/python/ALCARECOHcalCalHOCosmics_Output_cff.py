import FWCore.ParameterSet.Config as cms

#------------------------------------------------------
# Output block for HOCalibProducer
#-------------------------------------------------------
OutALCARECOHcalCalHOCosmics_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHOCosmics')
    ),
    outputCommands = cms.untracked.vstring(
        #'keep HOCalibVariabless_*_*_*'
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep HOCalibVariabless_hoCalibCosmicsProducer_*_*')
)


import copy
OutALCARECOHcalCalHOCosmics=copy.deepcopy(OutALCARECOHcalCalHOCosmics_noDrop)
OutALCARECOHcalCalHOCosmics.outputCommands.insert(0, "drop *")
