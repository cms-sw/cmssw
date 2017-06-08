import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using MinBias events
OutALCARECOAlCaPCCZeroBiasFromRECO_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOAlCaPCCZeroBiasFromRECO')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_*alcaPCCProducerZeroBias*_*_*')
)



import copy
OutALCARECOAlCaPCCZeroBiasFromRECO=copy.deepcopy(OutALCARECOAlCaPCCZeroBiasFromRECO_noDrop)
OutALCARECOAlCaPCCZeroBiasFromRECO.outputCommands.insert(0,"drop *")
