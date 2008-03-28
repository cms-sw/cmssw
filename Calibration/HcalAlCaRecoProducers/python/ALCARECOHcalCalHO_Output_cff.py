import FWCore.ParameterSet.Config as cms

#------------------------------------------------------
# Output block for HOCalibProducer
#-------------------------------------------------------
OutHOCalibProducer = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHO')
    ),
    outputCommands = cms.untracked.vstring('drop *', 'keep HOCalibVariabless_*_*_*')
)

