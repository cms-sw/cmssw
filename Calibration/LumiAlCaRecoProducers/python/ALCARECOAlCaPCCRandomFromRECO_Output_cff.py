import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using MinBias events
OutALCARECOAlCaPCCRandomFromRECO_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOAlCaPCCRandomFromRECO')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_*alcaPCCProducerRandom_*_*')
)



import copy
OutALCARECOAlCaPCCRandomFromRECO=copy.deepcopy(OutALCARECOAlCaPCCRandomFromRECO_noDrop)
OutALCARECOAlCaPCCRandomFromRECO.outputCommands.insert(0,"drop *")
