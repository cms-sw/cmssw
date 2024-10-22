import FWCore.ParameterSet.Config as cms

#------------------------------------------------------
# Output block for HOCalibProducer
#-------------------------------------------------------
OutALCARECOHcalCalHO_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHO')
    ),
    outputCommands = cms.untracked.vstring(
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep *_hoCalibProducer_*_*')
)


import copy
OutALCARECOHcalCalHO=copy.deepcopy(OutALCARECOHcalCalHO_noDrop)
OutALCARECOHcalCalHO.outputCommands.insert(0, "drop *")
