import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using MinBias events
OutALCARECOAlCaPCCZeroBias_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOAlCaPCCZeroBias')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_*alcaPCCProducer*_*_*') 
)



import copy
OutALCARECOAlCaPCCZeroBias=copy.deepcopy(OutALCARECOAlCaPCCZeroBias_noDrop)
OutALCARECOAlCaPCCZeroBias.outputCommands.insert(0,"drop *")
