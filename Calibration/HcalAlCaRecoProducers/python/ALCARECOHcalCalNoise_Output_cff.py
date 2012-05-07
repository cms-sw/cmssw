import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Noise

OutALCARECOHcalCalNoise_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalNoise')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_HcalNoiseProd_*_*',
	'keep edmTriggerResults_*_*_HLT'
)
)


import copy
OutALCARECOHcalCalNoise=copy.deepcopy(OutALCARECOHcalCalNoise_noDrop)
OutALCARECOHcalCalNoise.outputCommands.insert(0, "drop *")
