import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Noise

OutALCARECOHcalCalNoise = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalNoise')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_HcalNoiseProd_*_*',
	'keep edmTriggerResults_*_*_*'
)
)

